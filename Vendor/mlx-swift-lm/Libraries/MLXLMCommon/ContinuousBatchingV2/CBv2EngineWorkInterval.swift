// CBv2EngineWorkInterval.swift
//
// OS work interval for the engine step thread. Scheduling only: nothing here
// reads or writes model state, and any missing symbol or failed call leaves
// the engine running exactly as without it.

import Darwin
import Foundation

/// The engine thread joins an OS workgroup interval for each engine step, and
/// the step's host phase (readback, acceptance, next graph build and commit)
/// is one interval period with a short deadline, so the scheduler keeps the
/// thread on a performance core at full clock between GPU waits.
///
/// `AudioWorkIntervalCreate` is unavailable to Swift and AudioToolbox is not
/// linked, so every entry point is resolved with `dlsym`. Engine-queue only.
/// `DARKBLOOM_CBV2_ENGINE_WORK_INTERVAL=0` disables it.
final class CBv2EngineWorkInterval {
    private typealias Create = @convention(c) (UnsafePointer<CChar>, UInt32, OpaquePointer?) -> OpaquePointer?
    private typealias Join = @convention(c) (OpaquePointer, UnsafeMutableRawPointer) -> Int32
    private typealias Leave = @convention(c) (OpaquePointer, UnsafeMutableRawPointer) -> Void
    private typealias Start = @convention(c) (OpaquePointer, UInt64, UInt64, OpaquePointer?) -> Int32
    private typealias Finish = @convention(c) (OpaquePointer, OpaquePointer?) -> Int32
    private typealias Release = @convention(c) (OpaquePointer) -> Void

    /// The mach clock id from <os/clock.h>: interval stamps are in mach ticks.
    private static let machClock: UInt32 = 32
    private static let deadlineNanos: UInt64 = 4_000_000

    private let workgroup: OpaquePointer
    private let join: Join
    private let leave: Leave
    private let start: Start
    private let finish: Finish
    private let release: Release
    private let timebase: mach_timebase_info_data_t
    private let token: UnsafeMutableRawPointer
    private var joined = false
    private var started = false

    init?() {
        // RTLD_DEFAULT: libSystem's exports.
        let process = UnsafeMutableRawPointer(bitPattern: -2)
        var timebase = mach_timebase_info_data_t()
        guard ProcessInfo.processInfo.environment["DARKBLOOM_CBV2_ENGINE_WORK_INTERVAL"] != "0",
            let audio = dlopen(
                "/System/Library/Frameworks/AudioToolbox.framework/AudioToolbox", RTLD_LAZY | RTLD_LOCAL),
            let create = dlsym(audio, "AudioWorkIntervalCreate"),
            let join = dlsym(process, "os_workgroup_join"),
            let leave = dlsym(process, "os_workgroup_leave"),
            let start = dlsym(process, "os_workgroup_interval_start"),
            let finish = dlsym(process, "os_workgroup_interval_finish"),
            let release = dlsym(process, "os_release"),
            mach_timebase_info(&timebase) == KERN_SUCCESS, timebase.numer > 0, timebase.denom > 0,
            let workgroup = unsafeBitCast(create, to: Create.self)("com.eigen.cbv2.engine", Self.machClock, nil)
        else { return nil }
        self.workgroup = workgroup
        self.join = unsafeBitCast(join, to: Join.self)
        self.leave = unsafeBitCast(leave, to: Leave.self)
        self.start = unsafeBitCast(start, to: Start.self)
        self.finish = unsafeBitCast(finish, to: Finish.self)
        self.release = unsafeBitCast(release, to: Release.self)
        self.timebase = timebase
        // os_workgroup_join_token_s is 40 bytes on LP64.
        token = UnsafeMutableRawPointer.allocate(byteCount: 64, alignment: 16)
        token.initializeMemory(as: UInt8.self, repeating: 0, count: 64)
    }

    deinit {
        token.deallocate()
        release(workgroup)
    }

    /// Top of an engine step, on the engine thread.
    func stepBegan() {
        joined = join(workgroup, token) == 0
    }

    /// Opens the step's period at the readback stamp finalize already took.
    /// Starting from a thread that has not joined aborts the process, hence
    /// the `joined` guard.
    func hostWorkBegan(readbackNanos: UInt64) {
        guard joined, !started else { return }
        let begin = ticks(readbackNanos)
        started = start(workgroup, begin, begin &+ ticks(Self.deadlineNanos), nil) == 0
    }

    /// End of the engine step, on the same thread as `stepBegan`.
    func stepEnded() {
        if started {
            _ = finish(workgroup, nil)
            started = false
        }
        if joined {
            leave(workgroup, token)
            joined = false
        }
    }

    /// Floor of nanos * denom / numer without overflow; never later than the
    /// tick count the nanosecond stamp was derived from.
    private func ticks(_ nanos: UInt64) -> UInt64 {
        let numer = UInt64(timebase.numer)
        let denom = UInt64(timebase.denom)
        return (nanos / numer) * denom + (nanos % numer) * denom / numer
    }
}

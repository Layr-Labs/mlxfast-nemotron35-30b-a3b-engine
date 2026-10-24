import Foundation

/// A loaded drafter may keep its load-time compile traces on the same OS
/// thread as every engine that uses it. This holds no request or KV state.
public protocol CBv2EngineThreadProviding: AnyObject {
    var cbv2EngineThread: CBv2EngineThread { get }
}

/// FIFO executor with OS-thread affinity. The worker owns only its mailbox,
/// not this handle, so releasing the last owner wakes and retires the worker.
/// Existing queued work drains before exit; no join can block engine teardown.
public final class CBv2EngineThread: @unchecked Sendable {
    private typealias Work = @Sendable () -> Void

    private final class Mailbox: @unchecked Sendable {
        let condition = NSCondition()
        var pending: [Work?] = []
        var head = 0
        var closing = false

        func enqueue(_ work: @escaping Work) {
            condition.lock()
            precondition(!closing)
            pending.append(work)
            condition.signal()
            condition.unlock()
        }

        func next() -> Work? {
            condition.lock()
            defer { condition.unlock() }
            while head == pending.count && !closing { condition.wait() }
            guard head < pending.count else { return nil }
            let work = pending[head]
            pending[head] = nil
            head += 1
            if head == pending.count {
                pending.removeAll(keepingCapacity: true)
                head = 0
            }
            return work
        }

        func close() {
            condition.lock()
            closing = true
            condition.signal()
            condition.unlock()
        }

        func run() {
            while let work = next() {
                autoreleasepool(invoking: work)
            }
        }
    }

    /// Only the worker writes result/body; the semaphore publishes completion
    /// to the caller. Clear the escaped body BEFORE signaling its caller.
    private final class SyncCall<T, E: Error>: @unchecked Sendable {
        var body: (() throws(E) -> T)?
        var result: Result<T, E>?
        let done = DispatchSemaphore(value: 0)

        init(_ body: @escaping () throws(E) -> T) { self.body = body }

        private func invoke(_ body: () throws(E) -> T) -> Result<T, E> {
            do { return .success(try body()) } catch { return .failure(error) }
        }

        func run() {
            result = invoke(body!)
            body = nil
            done.signal()
        }
    }

    private let mailbox: Mailbox
    private let thread: Thread

    public init() {
        let mailbox = Mailbox()
        self.mailbox = mailbox
        let thread = Thread { mailbox.run() }
        thread.name = "com.eigen.cbv2.engine"
        thread.qualityOfService = .userInteractive
        self.thread = thread
        thread.start()
    }

    deinit { mailbox.close() }

    func async(execute work: @escaping @Sendable () -> Void) {
        mailbox.enqueue(work)
    }

    func asyncAfter(deadline: DispatchTime, execute work: @escaping @Sendable () -> Void) {
        // Only the timer callback uses GCD. The engine block still runs on
        // the worker; the loop's existing idle-generation guard is unchanged.
        DispatchQueue.global(qos: .userInteractive).asyncAfter(deadline: deadline) { [weak self] in
            self?.async(execute: work)
        }
    }

    /// Like serial DispatchQueue.sync, callers must not already be on this
    /// executor. Non-Sendable captures/results remain scoped to this barrier.
    public func sync<T, E: Error>(execute body: () throws(E) -> T) throws(E) -> T {
        precondition(Thread.current !== thread, "recursive engine-thread sync")
        let result: Result<T, E> = withoutActuallyEscaping(body) { escaped in
            let call = SyncCall(escaped)
            async { call.run() }
            call.done.wait()
            return call.result!
        }
        return try result.get()
    }
}

/// Preserve the existing per-engine GCD queue for drafters that do not opt in.
/// Every engine mutation, including synchronous control calls, uses one route.
final class CBv2EngineExecutor: @unchecked Sendable {
    private let pinned: CBv2EngineThread?
    private let queue: DispatchQueue?

    init(pinned: CBv2EngineThread?) {
        self.pinned = pinned
        self.queue = pinned == nil
            ? DispatchQueue(label: "com.eigen.cbv2.engine", qos: .userInteractive) : nil
    }

    func async(execute work: @escaping @Sendable () -> Void) {
        if let pinned { pinned.async(execute: work) }
        else { queue!.async(execute: work) }
    }

    func asyncAfter(deadline: DispatchTime, execute work: @escaping @Sendable () -> Void) {
        if let pinned { pinned.asyncAfter(deadline: deadline, execute: work) }
        else { queue!.asyncAfter(deadline: deadline, execute: work) }
    }

    func sync<T>(execute body: () throws -> T) rethrows -> T {
        if let pinned { return try pinned.sync(execute: body) }
        return try queue!.sync(execute: body)
    }
}

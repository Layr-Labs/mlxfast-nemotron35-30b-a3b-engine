// `phase_diagnostics` must cancel live free-run requests before it awaits
// engine shutdown. A begin that prefilled the seed but left decode rows
// in flight would otherwise sit behind the loop's shutdown drain.

import Foundation
import MLXLMCommon
import Testing

@testable import MLXRunners

@Suite("phase_diagnostics cancels live free-run work before shutdown")
struct BenchWorkerPhaseCloseCancelTests {
    private final class HeldContinuation: @unchecked Sendable {
        var continuation: AsyncStream<CBv2Event>.Continuation?
        func finish() {
            continuation?.finish()
            continuation = nil
        }
    }

    /// Shutdown waits until every submitted row is cancelled, like the real
    /// loop's shutdown drain when decode is still in flight.
    private final class StallUntilCancelEngine: CBv2Engine, @unchecked Sendable {
        private let queue = DispatchQueue(label: "StallUntilCancelEngine")
        private var live: Set<UInt64> = []
        private var held: [UInt64: HeldContinuation] = [:]
        private(set) var events: [String] = []

        func submit(_ request: CBv2Request) throws -> AsyncStream<CBv2Event> {
            let id = request.id.raw
            let start = MockTokens.freeRunStart(request.promptTokens)
            let holder = HeldContinuation()
            queue.sync {
                live.insert(id)
                held[id] = holder
            }
            return AsyncStream { continuation in
                holder.continuation = continuation
                continuation.yield(.delta(text: "", tokens: [start], logprobs: nil))
            }
        }

        func cancel(_ id: CBv2RequestID) {
            queue.sync {
                events.append("cancel:\(id.raw)")
                live.remove(id.raw)
                held.removeValue(forKey: id.raw)?.finish()
            }
        }

        func capacity() -> CBv2CapacitySnapshot {
            CBv2CapacitySnapshot(
                activeRequests: 0, waitingRequests: 0, kvBytesInUse: 0,
                kvBytesCapacity: 0, kvBytesBackendCapacity: 0, kvBytesReserved: 0,
                activeTokens: 0)
        }

        func shutdown() async {
            queue.sync { events.append("shutdown:start") }
            while true {
                let pending = queue.sync { live }
                if pending.isEmpty { break }
                try? await Task.sleep(for: .milliseconds(20))
            }
            queue.sync { events.append("shutdown:end") }
        }
    }

    private final class StallRunner: Runner, @unchecked Sendable {
        static var manifest: RunnerManifest { MockRunner.manifest }
        let engine = StallUntilCancelEngine()
        var servingModel: any LanguageModel { preconditionFailure() }
        var tokenizer: any MLXLMCommon.Tokenizer { preconditionFailure() }
        let eosTokenIDs: Set<Int> = []
        let layerKinds: [CBv2LayerKind] = []
        let loadedDecoders: [DecoderID] = [.serial, .mtp]
        let headProvenance: HeadProvenance? = nil
        let loadedModelType = "qwen4_exp_text"

        static func adopt(
            model: any LanguageModel,
            tokenizer: any MLXLMCommon.Tokenizer,
            configuration: ModelConfiguration,
            directory: URL,
            options: RunnerLoadOptions
        ) throws -> StallRunner {
            StallRunner()
        }

        func makeEngine(_ build: EngineBuild) throws -> any CBv2Engine { engine }
        func makeStepper() throws -> any TeacherForcedStepper { MockStepper() }
    }

    private func runPhaseClose(runner: StallRunner) async -> StallUntilCancelEngine {
        let transport = ScriptedTransport(lines: [
            "{\"id\":1,\"kind\":\"free_decode_begin\",\"seed_tokens\":[51,52,53]}",
            "{\"id\":2,\"kind\":\"phase_diagnostics\"}",
        ])
        let server = BenchWorkerServer(
            runner: runner,
            transport: transport,
            trusted: false,
            speculative: true,
            build: "fixture",
            device: "mock",
            kvBytesCapacity: 1 << 20,
            maxDecodeTokens: 64,
            memory: FixtureMemoryReporter(),
            nonce: "fixturenonce")
        await server.run()
        return runner.engine
    }

    @Test("phase_diagnostics cancels each live row before shutdown returns")
    func cancelBeforeShutdown() async throws {
        let runner = StallRunner()
        let engine = await runPhaseClose(runner: runner)
        let events = engine.events
        #expect(events.contains("cancel:1"))
        let cancelIndex = events.firstIndex(of: "cancel:1")
        let shutdownEnd = events.firstIndex(of: "shutdown:end")
        #expect(cancelIndex != nil)
        #expect(shutdownEnd != nil)
        #expect(cancelIndex! < shutdownEnd!)
    }
}

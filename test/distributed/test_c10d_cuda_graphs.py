# Owner(s): ["oncall: distributed"]

import contextlib
import sys
import unittest
from unittest import mock

import torch
import torch.distributed as dist


if not dist.is_available():
    print("distributed package not available, skipping tests", file=sys.stderr)
    sys.exit(0)

from c10d_backend_common import (
    C10dBackendTest,
    CUDA_BACKENDS,
    instantiate_backend_tests,
)

from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_distributed import (
    MultiProcessTestCase,
    skip_if_lt_x_gpu,
)
from torch.testing._internal.common_utils import parametrize, run_tests


ASYNC_OPS = (False, True)


class AbstractCUDAGraphsTest(C10dBackendTest):
    def _tensor(self, dtype, rank=None):
        rank = self.rank if rank is None else rank
        return torch.full((4,), rank + 1, dtype=torch.float32, device=self.device).to(
            dtype
        )

    def _capture_and_replay(self, op, async_op):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            work = op()
            if async_op:
                self.assertIsNotNone(work)
                work.wait()
            else:
                self.assertIsNone(work)
        graph.replay()
        torch.cuda.synchronize()

    def _test_all_reduce(self, dtype, async_op):
        tensor = self._tensor(dtype)
        self._capture_and_replay(
            lambda: dist.all_reduce(tensor, async_op=async_op),
            async_op,
        )
        expected = torch.full_like(tensor, sum(range(1, self.world_size + 1)))
        self.assertEqual(tensor, expected)

    def _test_broadcast(self, dtype, async_op):
        tensor = self._tensor(dtype)
        self._capture_and_replay(
            lambda: dist.broadcast(tensor, src=0, async_op=async_op),
            async_op,
        )
        self.assertEqual(tensor, self._tensor(dtype, rank=0))

    def _test_all_gather(self, dtype, async_op):
        tensor = self._tensor(dtype)
        output = [torch.empty_like(tensor) for _ in range(self.world_size)]
        self._capture_and_replay(
            lambda: dist.all_gather(output, tensor, async_op=async_op),
            async_op,
        )
        for rank, result in enumerate(output):
            self.assertEqual(result, self._tensor(dtype, rank))

    def _test_all_gather_single(self, dtype, async_op):
        tensor = self._tensor(dtype)
        output = torch.empty(
            self.world_size * tensor.numel(),
            dtype=dtype,
            device=self.device,
        )
        self._capture_and_replay(
            lambda: dist.all_gather_single(output, tensor, async_op=async_op),
            async_op,
        )
        expected = torch.cat(
            [self._tensor(dtype, rank) for rank in range(self.world_size)]
        )
        self.assertEqual(output, expected)

    def _test_reduce(self, dtype, async_op):
        tensor = self._tensor(dtype)
        self._capture_and_replay(
            lambda: dist.reduce(tensor, dst=0, async_op=async_op),
            async_op,
        )
        if self.rank == 0:
            expected = torch.full_like(tensor, sum(range(1, self.world_size + 1)))
            self.assertEqual(tensor, expected)

    def _test_gather(self, dtype, async_op):
        tensor = self._tensor(dtype)
        output = (
            [torch.empty_like(tensor) for _ in range(self.world_size)]
            if self.rank == 0
            else None
        )
        self._capture_and_replay(
            lambda: dist.gather(tensor, output, dst=0, async_op=async_op),
            async_op,
        )
        if self.rank == 0:
            for rank, result in enumerate(output):
                self.assertEqual(result, self._tensor(dtype, rank))

    def _test_scatter(self, dtype, async_op):
        output = torch.empty(4, dtype=dtype, device=self.device)
        inputs = (
            [self._tensor(dtype, rank) for rank in range(self.world_size)]
            if self.rank == 0
            else None
        )
        self._capture_and_replay(
            lambda: dist.scatter(output, inputs, src=0, async_op=async_op),
            async_op,
        )
        self.assertEqual(output, self._tensor(dtype, self.rank))

    def _test_reduce_scatter(self, dtype, async_op):
        inputs = [self._tensor(dtype) for _ in range(self.world_size)]
        output = torch.empty_like(inputs[0])
        self._capture_and_replay(
            lambda: dist.reduce_scatter(output, inputs, async_op=async_op),
            async_op,
        )
        expected = torch.full_like(output, sum(range(1, self.world_size + 1)))
        self.assertEqual(output, expected)

    def _test_reduce_scatter_single(self, dtype, async_op):
        inputs = torch.cat([self._tensor(dtype) for _ in range(self.world_size)])
        output = torch.empty(4, dtype=dtype, device=self.device)
        self._capture_and_replay(
            lambda: dist.reduce_scatter_single(output, inputs, async_op=async_op),
            async_op,
        )
        expected = torch.full_like(output, sum(range(1, self.world_size + 1)))
        self.assertEqual(output, expected)

    def _test_all_to_all(self, dtype, async_op):
        inputs = [self._tensor(dtype, self.rank) for _ in range(self.world_size)]
        outputs = [torch.empty_like(inputs[0]) for _ in range(self.world_size)]
        self._capture_and_replay(
            lambda: dist.all_to_all(outputs, inputs, async_op=async_op),
            async_op,
        )
        for rank, result in enumerate(outputs):
            self.assertEqual(result, self._tensor(dtype, rank))

    def _test_all_to_all_single(self, dtype, async_op):
        inputs = torch.cat(
            [self._tensor(dtype, self.rank) for _ in range(self.world_size)]
        )
        output = torch.empty_like(inputs)
        self._capture_and_replay(
            lambda: dist.all_to_all_single(output, inputs, async_op=async_op),
            async_op,
        )
        expected = torch.cat(
            [self._tensor(dtype, rank) for rank in range(self.world_size)]
        )
        self.assertEqual(output, expected)

    def test_collectives(self):
        self._init_pg()
        warmup = torch.ones(1, device=self.device)
        dist.all_reduce(warmup)
        torch.cuda.synchronize()

        tests = (
            self._test_all_reduce,
            self._test_broadcast,
            self._test_all_gather,
            self._test_all_gather_single,
            self._test_reduce,
            self._test_gather,
            self._test_scatter,
            self._test_reduce_scatter,
            self._test_reduce_scatter_single,
            self._test_all_to_all,
            self._test_all_to_all_single,
        )
        for test in tests:
            for dtype in self.dtypes:
                for async_op in ASYNC_OPS:
                    with self.subTest(
                        collective=test.__name__,
                        dtype=dtype,
                        async_op=async_op,
                    ):
                        test(dtype, async_op)

    def test_barrier(self):
        if not self.supports_cuda_graph_barrier:
            self.skipTest(f"{self.backend_name} barrier does not support CUDA graphs")
        self._init_pg()
        dist.all_reduce(torch.ones(1, device=self.device))
        torch.cuda.synchronize()
        for async_op in ASYNC_OPS:
            self._capture_and_replay(
                lambda: dist.barrier(
                    device_ids=[self.rank],
                    async_op=async_op,
                ),
                async_op,
            )

    def test_complex_collectives(self):
        self._init_pg()
        dist.all_reduce(torch.ones(1, device=self.device))
        torch.cuda.synchronize()
        for dtype in self.complex_dtypes:
            for async_op in ASYNC_OPS:
                for test in (
                    self._test_all_reduce,
                    self._test_broadcast,
                    self._test_reduce,
                ):
                    with self.subTest(
                        collective=test.__name__,
                        dtype=dtype,
                        async_op=async_op,
                    ):
                        test(dtype, async_op)


instantiate_backend_tests(
    globals(), "CUDAGraphs", AbstractCUDAGraphsTest, CUDA_BACKENDS
)


@unittest.skipIf(not dist.is_nccl_available(), "NCCL required")
class NCCLCaptureBoundaryTest(C10dBackendTest, MultiProcessTestCase):
    backend_name = "nccl-legacy"

    @contextlib.contextmanager
    def _release_legacy_graphs(self):
        # NCCL shutdown requires captured graphs to be released, even if a
        # legacy compiled callable remains cached after Dynamo reset.
        graphs = []
        graph_context = torch.cuda.graph

        def record_graph(graph, *args, **kwargs):
            graphs.append(graph)
            return graph_context(graph, *args, **kwargs)

        try:
            with mock.patch("torch.cuda.graph", record_graph):
                yield
        finally:
            for graph in graphs:
                graph.reset()

    @skip_if_lt_x_gpu(2)
    @parametrize("trees", [False, True])
    def test_graph_break(self, device, trees):
        from torch._dynamo.utils import counters
        from torch._inductor import config

        self._init_pg()
        group_name = dist.distributed_c10d._get_default_group().group_name

        def model(inp):
            pending = torch.ops._c10d_functional.all_reduce(inp, "sum", group_name)
            torch._dynamo.graph_break()
            waited = torch.ops._c10d_functional.wait_tensor(pending)
            return waited.view(4, 4)[:, :2] + 1

        torch._dynamo.reset()
        skips_before = counters["inductor"]["cudagraph_skips"]
        try:
            with (
                config.patch("triton.cudagraph_trees", trees),
                contextlib.nullcontext() if trees else self._release_legacy_graphs(),
            ):
                compiled = torch.compile(model, mode="reduce-overhead")
                for offset in range(6):
                    torch.compiler.cudagraph_mark_step_begin()
                    inp = torch.full(
                        (16,), self.rank + offset + 1.0, device=self.device
                    )
                    total = sum(rank + offset + 1 for rank in range(self.world_size))
                    self.assertEqual(
                        compiled(inp),
                        torch.full((4, 2), total + 1.0, device=self.device),
                    )
                self.assertEqual(counters["inductor"]["cudagraph_skips"], skips_before)
        finally:
            torch._dynamo.reset()

    @skip_if_lt_x_gpu(2)
    @parametrize("trees", [False, True])
    @parametrize("coalesced", [False, True])
    def test_eager_wait(self, device, trees, coalesced):
        from torch._dynamo.utils import counters
        from torch._inductor import config

        self._init_pg()
        group_name = dist.distributed_c10d._get_default_group().group_name

        def model(inp):
            if coalesced:
                pending = torch.ops._c10d_functional.all_reduce_coalesced(
                    [inp, inp + 1], "sum", group_name
                )
            else:
                pending = [
                    torch.ops._c10d_functional.all_reduce(inp, "sum", group_name)
                ]
            return [out.view(4, 4) for out in pending]

        torch._dynamo.reset()
        skips_before = counters["inductor"]["cudagraph_skips"]
        try:
            with (
                config.patch("triton.cudagraph_trees", trees),
                contextlib.nullcontext() if trees else self._release_legacy_graphs(),
            ):
                compiled = torch.compile(model, fullgraph=True, mode="reduce-overhead")
                for offset in range(6):
                    torch.compiler.cudagraph_mark_step_begin()
                    inp = torch.full(
                        (16,), self.rank + offset + 1.0, device=self.device
                    )
                    outputs = compiled(inp)
                    for index, output in enumerate(outputs):
                        waited = torch.ops._c10d_functional.wait_tensor(output)
                        total = sum(
                            rank + offset + index + 1 for rank in range(self.world_size)
                        )
                        self.assertEqual(waited, torch.full_like(waited, total))
                    self.assertEqual(
                        torch._C._distributed_c10d._get_work_registry_size(), 0
                    )
                self.assertEqual(counters["inductor"]["cudagraph_skips"], skips_before)
        finally:
            torch._dynamo.reset()

    @skip_if_lt_x_gpu(2)
    @parametrize("trees", [False, True])
    @parametrize("collective", ["all_gather", "reduce_scatter", "all_to_all"])
    def test_eager_wait_compat_namespace(self, device, trees, collective):
        from torch._dynamo.utils import counters
        from torch._inductor import config

        self._init_pg()
        group_name = dist.distributed_c10d._get_default_group().group_name
        ops = torch.ops._c10d_functional_autograd

        def model(inp):
            if collective == "all_gather":
                return ops.all_gather_into_tensor(inp, self.world_size, group_name)
            if collective == "reduce_scatter":
                return ops.reduce_scatter_tensor(
                    inp, "sum", self.world_size, group_name
                )
            splits = [8] * self.world_size
            return ops.all_to_all_single(inp, splits, splits, group_name)

        torch._dynamo.reset()
        skips_before = counters["inductor"]["cudagraph_skips"]
        try:
            with (
                config.patch("triton.cudagraph_trees", trees),
                contextlib.nullcontext() if trees else self._release_legacy_graphs(),
            ):
                compiled = torch.compile(model, fullgraph=True, mode="reduce-overhead")
                for offset in range(6):
                    torch.compiler.cudagraph_mark_step_begin()
                    inp = torch.full(
                        (8 * self.world_size,),
                        self.rank + offset + 1.0,
                        device=self.device,
                    )
                    pending = compiled(inp)
                    waited = torch.ops._c10d_functional.wait_tensor(pending)
                    if collective == "reduce_scatter":
                        total = sum(
                            rank + offset + 1 for rank in range(self.world_size)
                        )
                        expected = torch.full_like(waited, total)
                    else:
                        chunk = inp.numel() if collective == "all_gather" else 8
                        expected = torch.cat(
                            [
                                torch.full(
                                    (chunk,), rank + offset + 1.0, device=self.device
                                )
                                for rank in range(self.world_size)
                            ]
                        )
                    self.assertEqual(waited, expected)
                    self.assertEqual(
                        torch._C._distributed_c10d._get_work_registry_size(), 0
                    )
                self.assertEqual(counters["inductor"]["cudagraph_skips"], skips_before)
        finally:
            torch._dynamo.reset()

    @skip_if_lt_x_gpu(2)
    @parametrize("return_alias", [False, True])
    def test_mutated_input(self, device, return_alias):
        from torch._inductor.cudagraph_trees import cudagraphify_impl

        self._init_pg()
        group_name = dist.distributed_c10d._get_default_group().group_name

        def model(args):
            inp, alias = args
            args.clear()
            torch.ops._c10d_functional.all_reduce_(inp, "sum", group_name)
            return (alias.view(4, 4),) if return_alias else ()

        torch._dynamo.reset()
        try:
            inp = torch.ones(16, device=self.device)
            captured = cudagraphify_impl(
                model,
                [inp, inp],
                static_input_idxs=(0, 1),
                mutated_input_idxs=(0,),
                device_index=self.rank,
                is_backward=False,
                is_inference=True,
                stack_traces=[None] if return_alias else [],
            )
            for offset in range(6):
                torch.compiler.cudagraph_mark_step_begin()
                inp.fill_(self.rank + offset + 1)
                self.assertEqual(len(captured([inp, inp])), int(return_alias))
                waited = torch.ops._c10d_functional.wait_tensor(inp)
                total = sum(rank + offset + 1 for rank in range(self.world_size))
                self.assertEqual(waited, torch.full_like(waited, total))
                self.assertEqual(
                    torch._C._distributed_c10d._get_work_registry_size(), 0
                )
        finally:
            torch._dynamo.reset()

    @skip_if_lt_x_gpu(2)
    def test_compiled_mixed_static_dynamic_partitions(self, device):
        import ast

        from torch._dynamo.utils import counters
        from torch._inductor.utils import run_and_get_code

        self._init_pg()
        group_name = dist.distributed_c10d._get_default_group().group_name

        def model(inp, other, dynamic):
            pending = torch.ops._c10d_functional.all_gather_into_tensor(
                inp, self.world_size, group_name
            )
            ready = torch.ops._c10d_functional.all_reduce(other, "sum", group_name)
            ready = torch.ops._c10d_functional.wait_tensor(ready)
            middle = torch.ops._c10d_functional.all_reduce(
                dynamic + ready[0], "sum", group_name
            )
            middle = torch.ops._c10d_functional.wait_tensor(middle)
            # The data dependency prevents hoisting dynamic computation before
            # the first partition. Collective ordering keeps the second gather
            # after it, while the final wait operands retain static shapes.
            local = torch.ops._c10d_functional.all_gather_into_tensor(
                other, self.world_size, group_name
            )
            pending, local = torch.ops._c10d_functional.wait_tensors([pending, local])
            return pending + local + middle.sum()

        torch._dynamo.reset()
        skips_before = counters["inductor"]["cudagraph_skips"]
        try:
            compiled = torch.compile(
                model,
                fullgraph=True,
                options={
                    "triton.cudagraphs": True,
                    "graph_partition": True,
                    "triton.cudagraph_skip_dynamic_graphs": True,
                },
            )
            inp = torch.ones(16, device=self.device)
            other = torch.full_like(inp, 2)
            dynamic = torch.ones(8, device=self.device)
            torch._dynamo.mark_dynamic(dynamic, 0)
            _, code = run_and_get_code(compiled, inp, other, dynamic)
            source = "\n".join(code)
            partitions = {
                node.name: ast.get_source_segment(source, node)
                for node in ast.walk(ast.parse(source))
                if isinstance(node, ast.FunctionDef)
                and node.name.startswith("partition_")
            }
            self.assertEqual(set(partitions), {"partition_0", "partition_1"})
            self.assertIn("all_gather_into_tensor", partitions["partition_0"])
            self.assertIn("all_gather_into_tensor", partitions["partition_1"])
            self.assertIn("wait_tensors.default", partitions["partition_1"])
            # The dynamic reduction runs between the two captured partitions.
            self.assertRegex(
                source,
                r"(?s)self\.partitions\[0\].*\.run\(.*self\.partitions\[1\]",
            )
            with torch.compiler.set_stance("fail_on_recompile"):
                for offset, size in enumerate((8, 8, 16, 16, 8, 8)):
                    torch.compiler.cudagraph_mark_step_begin()
                    inp.fill_(self.rank + offset + 1)
                    other.fill_(self.rank + offset + 2)
                    dynamic = torch.full((size,), float(offset + 1), device=self.device)
                    torch._dynamo.mark_dynamic(dynamic, 0)
                    output = compiled(inp, other, dynamic)
                    expected = torch.cat(
                        [
                            torch.full_like(
                                inp,
                                2 * (rank + offset)
                                + 3
                                + self.world_size
                                * size
                                * (
                                    offset
                                    + 1
                                    + self.world_size * (offset + 2)
                                    + self.world_size * (self.world_size - 1) / 2
                                ),
                            )
                            for rank in range(self.world_size)
                        ]
                    )
                    self.assertEqual(output, expected)
            self.assertEqual(counters["inductor"]["cudagraph_skips"], skips_before)
        finally:
            torch._dynamo.reset()

    @skip_if_lt_x_gpu(2)
    def test_partition_intermediate_mutation(self, device):
        from torch._dynamo.utils import counters
        from torch._inductor.utils import run_and_get_code

        self._init_pg()
        group_name = dist.distributed_c10d._get_default_group().group_name

        def model(dynamic):
            # The dynamic reduction runs eagerly; the collective mutates its
            # intermediate result inside the following static capture region.
            pending = dynamic.sum().reshape(1)
            return torch.ops._c10d_functional.all_reduce(pending, "sum", group_name)

        torch._dynamo.reset()
        skips_before = counters["inductor"]["cudagraph_skips"]
        try:
            compiled = torch.compile(
                model,
                fullgraph=True,
                options={
                    "triton.cudagraphs": True,
                    "graph_partition": True,
                    "triton.cudagraph_skip_dynamic_graphs": True,
                },
            )
            dynamic = torch.ones(8, device=self.device)
            torch._dynamo.mark_dynamic(dynamic, 0)
            pending, code = run_and_get_code(compiled, dynamic)
            torch.ops._c10d_functional.wait_tensor(pending)
            source = "\n".join(code)
            self.assertEqual(source.count("def partition_"), 1)
            self.assertIn("all_reduce_.default", source)
            with torch.compiler.set_stance("fail_on_recompile"):
                for offset, size in enumerate((8, 8, 16, 16, 8, 8)):
                    torch.compiler.cudagraph_mark_step_begin()
                    dynamic = torch.full(
                        (size,), self.rank + offset + 1.0, device=self.device
                    )
                    torch._dynamo.mark_dynamic(dynamic, 0)
                    pending = compiled(dynamic)
                    output = torch.ops._c10d_functional.wait_tensor(pending)
                    expected = size * sum(
                        rank + offset + 1 for rank in range(self.world_size)
                    )
                    self.assertEqual(output, torch.full_like(output, expected))
            self.assertEqual(counters["inductor"]["cudagraph_skips"], skips_before)
        finally:
            torch._dynamo.reset()

    @skip_if_lt_x_gpu(2)
    @parametrize("trees", [False, True])
    def test_readonly_input_preserves_pending_work(self, device, trees):
        from torch._inductor.compile_fx import cudagraphify_impl as legacy_capture
        from torch._inductor.cudagraph_trees import cudagraphify_impl as tree_capture

        self._init_pg()
        group_name = dist.distributed_c10d._get_default_group().group_name

        def model(args):
            inp = args[0]
            args.clear()
            return (inp.view(4, 4),)

        torch._dynamo.reset()
        try:
            inp = torch.full((16,), self.rank + 1.0, device=self.device)
            torch.ops._c10d_functional.all_reduce_(inp, "sum", group_name)
            if trees:
                captured = tree_capture(
                    model,
                    [inp],
                    static_input_idxs=(0,),
                    device_index=self.rank,
                    is_backward=False,
                    is_inference=True,
                    stack_traces=[None],
                )
            else:
                captured = legacy_capture(model, [inp], static_input_idxs=(0,))
            for offset in range(6):
                torch.compiler.cudagraph_mark_step_begin()
                if offset:
                    inp.fill_(self.rank + offset + 1)
                    torch.ops._c10d_functional.all_reduce_(inp, "sum", group_name)
                (output,) = captured([inp])
                # Returning an input view must not consume work launched outside
                # this capture, even if synchronization has already completed it.
                self.assertEqual(
                    torch._C._distributed_c10d._get_work_registry_size(), 1
                )
                waited = torch.ops._c10d_functional.wait_tensor(output)
                total = sum(rank + offset + 1 for rank in range(self.world_size))
                self.assertEqual(waited, torch.full_like(waited, total))
        finally:
            torch._dynamo.reset()


instantiate_device_type_tests(NCCLCaptureBoundaryTest, globals(), only_for="cuda")


if __name__ == "__main__":
    run_tests()

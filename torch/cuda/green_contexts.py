from collections.abc import Callable

import torch


__all__ = [
    "GreenContext",
    "is_localization_supported",
    "execute_in_green_contexts",
]

_GreenContext = object
_LocalizedGreenContextMemPool = object
_LocalizedAllocator = object
SUPPORTED = False

if (
    hasattr(torch._C, "_CUDAGreenContext")
    and hasattr(torch._C, "_CUDALocalizedGreenContextMemPool")
    and hasattr(torch._C, "_CUDALocalizedAllocator")
    and hasattr(torch._C, "_get_num_memory_nodes")
):
    _GreenContext = torch._C._CUDAGreenContext  # type: ignore[misc]
    _LocalizedGreenContextMemPool = torch._C._CUDALocalizedGreenContextMemPool  # type: ignore[misc]
    _LocalizedAllocator = torch._C._CUDALocalizedAllocator  # type: ignore[misc]
    SUPPORTED = True


def is_localization_supported(device_id: int) -> bool:
    r"""Return a bool indicating if the current CUDA/ROCm device supports green context localization."""
    if not SUPPORTED:
        return False
    if torch.version.hip:
        return False
    if not torch.cuda.is_available() or torch.version.cuda is None:
        return False
    major, minor = map(int, torch.version.cuda.split("."))
    if major < 13 or (major == 13 and minor < 4):
        return False
    # pyrefly: ignore [missing-attribute]
    return torch._C._get_num_memory_nodes(device_id) > 1


# Python shim helps Sphinx process docstrings more reliably.
# pyrefly: ignore [invalid-inheritance]
class GreenContext(_GreenContext):
    r"""Wrapper around a CUDA green context.

    .. warning::
       This API is in beta and may change in future releases.
    """

    @staticmethod
    def create(num_sms: int, device_id: int = 0) -> _GreenContext:
        r"""Create a CUDA green context.

        Arguments:
            num_sms (int): The number of SMs to use in the green context.
            device_id (int, optional): The device index of green context.
        """
        if not SUPPORTED:
            raise RuntimeError("PyTorch was not built with Green Context support!")
        return _GreenContext.create(num_sms, device_id)  # type: ignore[attr-defined]

    @staticmethod
    def create_localized(device_id: int = 0) -> list[_GreenContext]:
        r"""Create a CUDA green context for each memory node on the device.

        Arguments:
            device_id (int, optional): The device index of green context.
        """
        if not is_localization_supported(device_id):
            raise RuntimeError(
                "Green Context localization is not supported on this device!"
            )
        return _GreenContext.create_localized(device_id)  # type: ignore[attr-defined]

    # Note that these functions are bypassed but we define them here
    # for Sphinx documentation purposes
    def set_context(self, block_current_stream: bool = True) -> None:  # pylint: disable=useless-parent-delegation
        r"""Make the green context the current context.

        This blocks execution of the default green context stream to wait on
        in-flight operations on the current stream, unless `block_current_stream` is set
        to ``False``. Note that setting `block_current_stream` to ``False`` is generally
        unsafe, unless manual synchronization between the streams is performed, as in
        `execute_in_green_contexts`. It is thus recommended to use
        `execute_in_green_contexts` for most use-cases instead.

        Arguments:
            block_current_stream (bool, optional): Whether to block the current stream.
                                                   Default is ``True``.
        """
        return super().set_context(block_current_stream)  # type: ignore[misc]

    def pop_context(self, block_parent_stream: bool = True) -> None:  # pylint: disable=useless-parent-delegation
        r"""Assuming the green context is the current context, pop it from the
        context stack and restore the previous context.

        This blocks execution of the parent stream to wait for current operations on
        the green context's stream. The parent stream is the non-green context
        stream set before the green context was activated.

        This behavior can be disabled by setting `block_parent_stream` to ``False``,
        but this is generally considered unsafe, unless manual synchronization between
        the streams is performed, as in `execute_in_green_contexts`. It is thus recommended
        to use `execute_in_green_contexts` for most use-cases instead.

        Arguments:
            block_parent_stream (bool, optional): Whether to block the parent stream.
                                                  Default is ``True``.
        """
        return super().pop_context(block_parent_stream)  # type: ignore[misc]

    def Stream(self) -> torch.Stream:
        r"""Return the CUDA Stream used by the green context."""
        return super().Stream()


def execute_in_green_contexts(
    green_contexts: list[GreenContext], fn: Callable[[int, GreenContext], None]
) -> None:
    r"""Execute a function in a list of green contexts, in parallel.

    The streams are set up in such a way that minimal blocking is used while
    correctly setting and popping the green contexts: this implements a so-called
    diamond pattern where execution of an asynchronous function is performed in
    the default streams of several green contexts. Once all "local" green context
    function calls have been issued to their respective default stream,
    execution is "joined" back into the current stream.

    Arguments:
        green_contexts (list[GreenContext]): The list of green contexts to use.
        fn (Callable): The function to execute.
          It is called with the index and the value of the current green context.
    """
    if not green_contexts:
        raise ValueError("Need at least one green context to execute in!")
    if len(green_contexts) == 1:
        green_contexts[0].set_context(block_current_stream=True)
        try:
            fn(0, green_contexts[0])
        finally:
            green_contexts[0].pop_context(block_parent_stream=True)
        return

    green_events = [torch.cuda.Event() for _ in green_contexts]
    # first record an event in current stream, on which all green context
    # streams need to wait
    main_event = torch.cuda.Event()
    main_event.record()
    for i, green_context in enumerate(green_contexts):
        # the green context shouldn't block on current stream here,
        # but it needs to block on the main stream event
        green_context.set_context(block_current_stream=False)
        main_event.wait()
        # execute the function in the green context
        fn(i, green_context)
        # record a separate event for each green context stream
        green_events[i].record()
        # now pop the context without blocking
        green_context.pop_context(block_parent_stream=False)

    # at this point, all functions were executed, and the main stream
    # needs to block on all green context streams to "join" them
    for green_event in green_events:
        green_event.wait()

from math import sqrt
from typing import List, Optional, Union, Any, Callable
import warnings

import torch
from torch import fx

import e3nn
from e3nn import o3
from e3nn.util import prod
from e3nn.util.codegen import CodeGenMixin
from e3nn.util.jit import compile_mode
from e3nn.o3._tensor_product._codegen import codegen_tensor_product_left_right, codegen_tensor_product_right
# from ._instruction import Instruction
from e3nn.o3._tensor_product._instruction import Instruction
from torch import nn

import copy
import math
from torch.nn import Linear
from .so3rotation import SO3_Rotation
from ..equiformer_v2.so3 import SO3_Embedding
from ..equiformer_v2.radial_function import RadialFunction
from e3nn.o3 import Linear

# A list, in order of priority, of codegen providers for the tensor product.
# If a provider does not support the parameters it is given, it should
# return `None`, in which case the next provider in the list will be tried.
_CODEGEN_PROVIDERS_LEFT_RIGHT: List[Callable] = [codegen_tensor_product_left_right]
_CODEGEN_PROVIDERS_RIGHT: List[Callable] = [codegen_tensor_product_right]


@compile_mode("script")
class Sparse_TensorProduct(CodeGenMixin, torch.nn.Module):
    r"""Tensor product with parametrized paths.

    Parameters
    ----------
    irreps_in1 : `e3nn.o3.Irreps`
        Irreps for the first input.

    irreps_in2 : `e3nn.o3.Irreps`
        Irreps for the second input.

    irreps_out : `e3nn.o3.Irreps`
        Irreps for the output.

    instructions : list of tuple
        List of instructions ``(i_1, i_2, i_out, mode, train[, path_weight])``.

        Each instruction puts ``in1[i_1]`` :math:`\otimes` ``in2[i_2]`` into ``out[i_out]``.

        * ``mode``: `str`. Determines the way the multiplicities are treated, ``"uvw"`` is fully connected. Other valid
        options are: ``'uvw'``, ``'uvu'``, ``'uvv'``, ``'uuw'``, ``'uuu'``, and ``'uvuv'``.
        * ``train``: `bool`. `True` if this path should have learnable weights, otherwise `False`.
        * ``path_weight``: `float`. A fixed multiplicative weight to apply to the output of this path. Defaults to 1. Note
        that setting ``path_weight`` breaks the normalization derived from ``in1_var``/``in2_var``/``out_var``.

    in1_var : list of float, Tensor, or None
        Variance for each irrep in ``irreps_in1``. If ``None``, all default to ``1.0``.

    in2_var : list of float, Tensor, or None
        Variance for each irrep in ``irreps_in2``. If ``None``, all default to ``1.0``.

    out_var : list of float, Tensor, or None
        Variance for each irrep in ``irreps_out``. If ``None``, all default to ``1.0``.

    irrep_normalization : {'component', 'norm'}
        The assumed normalization of the input and output representations. If it is set to "norm":

        .. math::

            \| x \| = \| y \| = 1 \Longrightarrow \| x \otimes y \| = 1

    path_normalization : {'element', 'path'}
        If set to ``element``, each output is normalized by the total number of elements (independently of their paths).
        If it is set to ``path``, each path is normalized by the total number of elements in the path, then each output is
        normalized by the number of paths.

    internal_weights : bool
        whether the `e3nn.o3.TensorProduct` contains its learnable weights as a parameter

    shared_weights : bool
        whether the learnable weights are shared among the input's extra dimensions

        * `True` :math:`z_i = w x_i \otimes y_i`
        * `False` :math:`z_i = w_i x_i \otimes y_i`

        where here :math:`i` denotes a *batch-like* index.
        ``shared_weights`` cannot be `False` if ``internal_weights`` is `True`.

    compile_left_right : bool
        whether to compile the forward function, true by default

    compile_right : bool
        whether to compile the ``.right`` function, false by default

    Examples
    --------
    Create a module that computes elementwise the cross-product of 16 vectors with 16 vectors :math:`z_u = x_u \wedge y_u`

    >>> module = TensorProduct(
    ...     "16x1o", "16x1o", "16x1e",
    ...     [
    ...         (0, 0, 0, "uuu", False)
    ...     ]
    ... )

    Now mix all 16 vectors with all 16 vectors to makes 16 pseudo-vectors :math:`z_w = \sum_{u,v} w_{uvw} x_u \wedge y_v`

    >>> module = TensorProduct(
    ...     [(16, (1, -1))],
    ...     [(16, (1, -1))],
    ...     [(16, (1,  1))],
    ...     [
    ...         (0, 0, 0, "uvw", True)
    ...     ]
    ... )

    With custom input variance and custom path weights:

    >>> module = TensorProduct(
    ...     "8x0o + 8x1o",
    ...     "16x1o",
    ...     "16x1e",
    ...     [
    ...         (0, 0, 0, "uvw", True, 3),
    ...         (1, 0, 0, "uvw", True, 1),
    ...     ],
    ...     in2_var=[1/16]
    ... )

    Example of a dot product:

    >>> irreps = o3.Irreps("3x0e + 4x0o + 1e + 2o + 3o")
    >>> module = TensorProduct(irreps, irreps, "0e", [
    ...     (i, i, 0, 'uuw', False)
    ...     for i, (mul, ir) in enumerate(irreps)
    ... ])

    Implement :math:`z_u = x_u \otimes (\sum_v w_{uv} y_v)`

    >>> module = TensorProduct(
    ...     "8x0o + 7x1o + 3x2e",
    ...     "10x0e + 10x1e + 10x2e",
    ...     "8x0o + 7x1o + 3x2e",
    ...     [
    ...         # paths for the l=0:
    ...         (0, 0, 0, "uvu", True),  # 0x0->0
    ...         # paths for the l=1:
    ...         (1, 0, 1, "uvu", True),  # 1x0->1
    ...         (1, 1, 1, "uvu", True),  # 1x1->1
    ...         (1, 2, 1, "uvu", True),  # 1x2->1
    ...         # paths for the l=2:
    ...         (2, 0, 2, "uvu", True),  # 2x0->2
    ...         (2, 1, 2, "uvu", True),  # 2x1->2
    ...         (2, 2, 2, "uvu", True),  # 2x2->2
    ...     ]
    ... )

    Tensor Product using the xavier uniform initialization:

    >>> irreps_1 = o3.Irreps("5x0e + 10x1o + 1x2e")
    >>> irreps_2 = o3.Irreps("5x0e + 10x1o + 1x2e")
    >>> irreps_out = o3.Irreps("5x0e + 10x1o + 1x2e")
    >>> # create a Fully Connected Tensor Product
    >>> module = o3.TensorProduct(
    ...     irreps_1,
    ...     irreps_2,
    ...     irreps_out,
    ...     [
    ...         (i_1, i_2, i_out, "uvw", True, mul_1 * mul_2)
    ...         for i_1, (mul_1, ir_1) in enumerate(irreps_1)
    ...         for i_2, (mul_2, ir_2) in enumerate(irreps_2)
    ...         for i_out, (mul_out, ir_out) in enumerate(irreps_out)
    ...         if ir_out in ir_1 * ir_2
    ...     ]
    ... )
    >>> with torch.no_grad():
    ...     for weight in module.weight_views():
    ...         mul_1, mul_2, mul_out = weight.shape
    ...         # formula from torch.nn.init.xavier_uniform_
    ...         a = (6 / (mul_1 * mul_2 + mul_out))**0.5
    ...         new_weight = torch.empty_like(weight)
    ...         new_weight.uniform_(-a, a)
    ...         weight[:] = new_weight
    tensor(...)
    >>> n = 1_000
    >>> vars = module(irreps_1.randn(n, -1), irreps_2.randn(n, -1)).var(0)
    >>> assert vars.min() > 1 / 3
    >>> assert vars.max() < 3
    """
    instructions: List[Any]
    shared_weights: bool
    internal_weights: bool
    weight_numel: int
    _specialized_code: bool
    _optimize_einsums: bool
    _profiling_str: str
    _in1_dim: int
    _in2_dim: int

    def __init__(
        self,
        irreps_in1: o3.Irreps,
        irreps_in2: o3.Irreps,
        irreps_out: o3.Irreps,
        instructions: List[tuple],
        in1_var: Optional[Union[List[float], torch.Tensor]] = None,
        in2_var: Optional[Union[List[float], torch.Tensor]] = None,
        out_var: Optional[Union[List[float], torch.Tensor]] = None,
        irrep_normalization: str = None,
        path_normalization: str = None,
        internal_weights: Optional[bool] = None,
        shared_weights: Optional[bool] = None,
        compile_left_right: bool = True,
        compile_right: bool = False,
        normalization=None,  # for backward compatibility
        _specialized_code: Optional[bool] = None,
        _optimize_einsums: Optional[bool] = None,
    ):
        # === Setup ===
        super().__init__()

        if normalization is not None:
            warnings.warn("`normalization` is deprecated. Use `irrep_normalization` instead.", DeprecationWarning)
            irrep_normalization = normalization

        if irrep_normalization is None:
            irrep_normalization = "component"

        if path_normalization is None:
            path_normalization = "element"

        assert irrep_normalization in ["component", "norm", "none"]
        assert path_normalization in ["element", "path", "none"]

        self.irreps_in1 = o3.Irreps(irreps_in1)
        self.irreps_in2 = o3.Irreps(irreps_in2)
        self.irreps_out = o3.Irreps(irreps_out)
        del irreps_in1, irreps_in2, irreps_out

        instructions = [x if len(x) == 6 else x + (1.0,) for x in instructions]
        instructions = [
            Instruction(
                i_in1=i_in1,
                i_in2=i_in2,
                i_out=i_out,
                connection_mode=connection_mode,
                has_weight=has_weight,
                path_weight=path_weight,
                path_shape={
                    "uvw": (self.irreps_in1[i_in1].mul, self.irreps_in2[i_in2].mul, self.irreps_out[i_out].mul),
                    "uvu": (self.irreps_in1[i_in1].mul, self.irreps_in2[i_in2].mul),
                    "uvv": (self.irreps_in1[i_in1].mul, self.irreps_in2[i_in2].mul),
                    "uuw": (self.irreps_in1[i_in1].mul, self.irreps_out[i_out].mul),
                    "uuu": (self.irreps_in1[i_in1].mul,),
                    "uvuv": (self.irreps_in1[i_in1].mul, self.irreps_in2[i_in2].mul),
                    "uvu<v": (self.irreps_in1[i_in1].mul * (self.irreps_in2[i_in2].mul - 1) // 2,),
                    "u<vw": (self.irreps_in1[i_in1].mul * (self.irreps_in2[i_in2].mul - 1) // 2, self.irreps_out[i_out].mul),
                }[connection_mode],
            )
            for i_in1, i_in2, i_out, connection_mode, has_weight, path_weight in instructions
        ]

        if in1_var is None:
            in1_var = [1.0 for _ in range(len(self.irreps_in1))]
        else:
            in1_var = [float(var) for var in in1_var]
            assert len(in1_var) == len(self.irreps_in1), "Len of ir1_var must be equal to len(irreps_in1)"

        if in2_var is None:
            in2_var = [1.0 for _ in range(len(self.irreps_in2))]
        else:
            in2_var = [float(var) for var in in2_var]
            assert len(in2_var) == len(self.irreps_in2), "Len of ir2_var must be equal to len(irreps_in2)"

        if out_var is None:
            out_var = [1.0 for _ in range(len(self.irreps_out))]
        else:
            out_var = [float(var) for var in out_var]
            assert len(out_var) == len(self.irreps_out), "Len of out_var must be equal to len(irreps_out)"

        def num_elements(ins):
            return {
                "uvw": (self.irreps_in1[ins.i_in1].mul * self.irreps_in2[ins.i_in2].mul),
                "uvu": self.irreps_in2[ins.i_in2].mul,
                "uvv": self.irreps_in1[ins.i_in1].mul,
                "uuw": self.irreps_in1[ins.i_in1].mul,
                "uuu": 1,
                "uvuv": 1,
                "uvu<v": 1,
                "u<vw": self.irreps_in1[ins.i_in1].mul * (self.irreps_in2[ins.i_in2].mul - 1) // 2,
            }[ins.connection_mode]

        normalization_coefficients = []
        for ins in instructions:
            mul_ir_in1 = self.irreps_in1[ins.i_in1]
            mul_ir_in2 = self.irreps_in2[ins.i_in2]
            mul_ir_out = self.irreps_out[ins.i_out]
            assert mul_ir_in1.ir.p * mul_ir_in2.ir.p == mul_ir_out.ir.p
            assert abs(mul_ir_in1.ir.l - mul_ir_in2.ir.l) <= mul_ir_out.ir.l <= mul_ir_in1.ir.l + mul_ir_in2.ir.l
            assert ins.connection_mode in ["uvw", "uvu", "uvv", "uuw", "uuu", "uvuv", "uvu<v", "u<vw"]

            if irrep_normalization == "component":
                alpha = mul_ir_out.ir.dim
            if irrep_normalization == "norm":
                alpha = mul_ir_in1.ir.dim * mul_ir_in2.ir.dim
            if irrep_normalization == "none":
                alpha = 1

            if path_normalization == "element":
                x = sum(in1_var[i.i_in1] * in2_var[i.i_in2] * num_elements(i) for i in instructions if i.i_out == ins.i_out)
            if path_normalization == "path":
                x = in1_var[ins.i_in1] * in2_var[ins.i_in2] * num_elements(ins)
                x *= len([i for i in instructions if i.i_out == ins.i_out])
            if path_normalization == "none":
                x = 1

            if x > 0.0:
                alpha /= x

            alpha *= out_var[ins.i_out]
            alpha *= ins.path_weight

            normalization_coefficients += [sqrt(alpha)]

        self.instructions = [
            Instruction(ins.i_in1, ins.i_in2, ins.i_out, ins.connection_mode, ins.has_weight, alpha, ins.path_shape)
            for ins, alpha in zip(instructions, normalization_coefficients)
        ]

        self._in1_dim = self.irreps_in1.dim
        self._in2_dim = self.irreps_in2.dim

        if shared_weights is False and internal_weights is None:
            internal_weights = False

        if shared_weights is None:
            shared_weights = True

        if internal_weights is None:
            internal_weights = shared_weights and any(i.has_weight for i in self.instructions)

        assert shared_weights or not internal_weights
        self.internal_weights = internal_weights
        self.shared_weights = shared_weights

        opt_defaults = e3nn.get_optimization_defaults()
        self._specialized_code = _specialized_code if _specialized_code is not None else opt_defaults["specialized_code"]
        self._optimize_einsums = _optimize_einsums if _optimize_einsums is not None else opt_defaults["optimize_einsums"]
        del opt_defaults

        # Generate the actual tensor product code
        if compile_left_right:
            for codegen in _CODEGEN_PROVIDERS_LEFT_RIGHT:
                graphmod_left_right = codegen(
                    self.irreps_in1,
                    self.irreps_in2,
                    self.irreps_out,
                    self.instructions,
                    self.shared_weights,
                    self._specialized_code,
                    self._optimize_einsums,
                )
                if graphmod_left_right is not None:
                    break
            assert graphmod_left_right is not None
        else:
            graphmod_left_right = fx.Graph()
            graphmod_left_right.placeholder("x1", torch.Tensor)
            graphmod_left_right.placeholder("x2", torch.Tensor)
            graphmod_left_right.placeholder("w", torch.Tensor)
            graphmod_left_right.placeholder("path_mask",torch.Tensor)
            graphmod_left_right.call_function(
                torch._assert,
                args=(
                    False,
                    "`left_right` method is not compiled, set `compile_left_right` to True when creating the TensorProduct",
                ),
            )
            graphmod_left_right = fx.GraphModule(torch.nn.Module(), graphmod_left_right, class_name="tp_forward")

        if compile_right:
            for codegen in _CODEGEN_PROVIDERS_RIGHT:
                graphmod_right = codegen(
                    self.irreps_in1,
                    self.irreps_in2,
                    self.irreps_out,
                    self.instructions,
                    self.shared_weights,
                    self._specialized_code,
                    self._optimize_einsums,
                )
                if graphmod_right is not None:
                    break
            assert graphmod_right is not None
        else:
            graphmod_right = fx.Graph()
            graphmod_right.placeholder("x2", torch.Tensor)
            graphmod_right.placeholder("w", torch.Tensor)
            graphmod_right.call_function(
                torch._assert,
                args=(False, "`right` method is not compiled, set `compile_right` to True when creating the TensorProduct"),
            )
            graphmod_right = fx.GraphModule(torch.nn.Module(), graphmod_right, class_name="tp_forward")

        self._codegen_register({"_compiled_main_left_right": graphmod_left_right, "_compiled_main_right": graphmod_right})

        # === Determine weights ===
        self.weight_numel = sum(prod(ins.path_shape) for ins in self.instructions if ins.has_weight)

        if internal_weights and self.weight_numel > 0:
            assert self.shared_weights, "Having internal weights impose shared weights"
            self.weight = torch.nn.Parameter(torch.randn(self.weight_numel))
        else:
            # For TorchScript, there always has to be some kind of defined .weight
            self.register_buffer("weight", torch.Tensor())

        if self.irreps_out.dim > 0:
            output_mask = torch.cat(
                [
                    torch.ones(mul * ir.dim)
                    if any(
                        (ins.i_out == i_out) and (ins.path_weight != 0) and (0 not in ins.path_shape)
                        for ins in self.instructions
                    )
                    else torch.zeros(mul * ir.dim)
                    for i_out, (mul, ir) in enumerate(self.irreps_out)
                ]
            )
        else:
            output_mask = torch.ones(0)
        self.register_buffer("output_mask", output_mask)

        # For TorchScript, this needs to be done in advance:
        self._profiling_str = str(self)

        self.path_layer = nn.Linear(self.irreps_in1[0][0]+self.irreps_in2[0][0], len(self.instructions))

        self.tp_times = 1
        self.sparsity = 0.007
        self.er = (self.sparsity)*0.4

    def __repr__(self):
        npath = sum(prod(i.path_shape) for i in self.instructions)
        return (
            f"{self.__class__.__name__}"
            f"({self.irreps_in1.simplify()} x {self.irreps_in2.simplify()} "
            f"-> {self.irreps_out.simplify()} | {npath} paths | {self.weight_numel} weights)"
        )

    @torch.jit.unused
    def _prep_weights_python(self, weight: Optional[Union[torch.Tensor, List[torch.Tensor]]]) -> Optional[torch.Tensor]:
        if isinstance(weight, list):
            weight_shapes = [ins.path_shape for ins in self.instructions if ins.has_weight]
            if not self.shared_weights:
                weight = [w.reshape(-1, prod(shape)) for w, shape in zip(weight, weight_shapes)]
            else:
                weight = [w.reshape(prod(shape)) for w, shape in zip(weight, weight_shapes)]
            return torch.cat(weight, dim=-1)
        else:
            return weight

    def _get_weights(self, weight: Optional[torch.Tensor]) -> torch.Tensor:
        if not torch.jit.is_scripting():
            # If we're not scripting, then we're in Python and `weight` could be a List[Tensor]
            # deal with that:
            weight = self._prep_weights_python(weight)
        if weight is None:
            if self.weight_numel > 0 and not self.internal_weights:
                raise RuntimeError("Weights must be provided when the TensorProduct does not have `internal_weights`")
            return self.weight
        else:
            if self.shared_weights:
                assert weight.shape == (self.weight_numel,), "Invalid weight shape"
            else:
                assert weight.shape[-1] == self.weight_numel, "Invalid weight shape"
                assert weight.ndim > 1, "When shared weights is false, weights must have batch dimension"
            return weight

    @torch.jit.export
    def right(self, y, weight: Optional[torch.Tensor] = None):
        r"""Partially evaluate :math:`w x \otimes y`.

        It returns an operator in the form of a tensor that can act on an arbitrary :math:`x`.

        For example, if the tensor product above is expressed as

        .. math::

            w_{ijk} x_i y_j \rightarrow z_k

        then the right method returns a tensor :math:`b_{ik}` such that

        .. math::

            w_{ijk} y_j \rightarrow b_{ik}
            x_i b_{ik} \rightarrow z_k

        The result of this method can be applied with a tensor contraction:

        .. code-block:: python

            torch.einsum("...ik,...i->...k", right, input)

        Parameters
        ----------
        y : `torch.Tensor`
            tensor of shape ``(..., irreps_in2.dim)``

        weight : `torch.Tensor` or list of `torch.Tensor`, optional
            required if ``internal_weights`` is ``False``
            tensor of shape ``(self.weight_numel,)`` if ``shared_weights`` is ``True``
            tensor of shape ``(..., self.weight_numel)`` if ``shared_weights`` is ``False``
            or list of tensors of shapes ``weight_shape`` / ``(...) + weight_shape``.
            Use ``self.instructions`` to know what are the weights used for.

        Returns
        -------
        `torch.Tensor`
            tensor of shape ``(..., irreps_in1.dim, irreps_out.dim)``
        """
        assert y.shape[-1] == self._in2_dim, "Incorrect last dimension for y"

        # - PROFILER - with torch.autograd.profiler.record_function(self._profiling_str):
        real_weight = self._get_weights(weight)
        return self._compiled_main_right(y, real_weight)


    def _sum_tensors(self, xs: List[torch.Tensor], shape: torch.Size, like: torch.Tensor):
        if len(xs) > 0:
            out = xs[0]
            for x in xs[1:]:
                out = out + x
            return out
        return like.new_zeros(shape)

    def _handwrite_main_left_right(self,x1s,x2s,weights,instructions):
        # = Function definitions =
        empty = torch.empty((), device="cpu")  

        if self.shared_weights:
            output_shape = torch.broadcast_tensors(empty.expand(x1s.shape[:-1]), empty.expand(x2s.shape[:-1]))[0].shape
        else:
            output_shape = torch.broadcast_tensors(
                empty.expand(x1s.shape[:-1]), empty.expand(x2s.shape[:-1]), empty.expand(weights.shape[:-1])
            )[0].shape
        del empty

        # = Short-circut for zero dimensional =
        # We produce no code for empty instructions
        # instructions = [ins for ins in instructions if 0 not in ins.path_shape]

        # = Broadcast inputs =
        if self.shared_weights:
            x1s, x2s = x1s.broadcast_to(output_shape + (-1,)), x2s.broadcast_to(output_shape + (-1,))
        else:
            x1s, x2s, weights = (
                x1s.broadcast_to(output_shape + (-1,)),
                x2s.broadcast_to(output_shape + (-1,)),
                weights.broadcast_to(output_shape + (-1,)),
            )

        output_shape = output_shape + (self.irreps_out.dim,)

        x1s = x1s.reshape(-1, self.irreps_in1.dim)
        x2s = x2s.reshape(-1, self.irreps_in2.dim)

        batch_numel = x1s.shape[0]

        # = Determine number of weights and reshape weights ==
        weight_numel = sum(prod(ins.path_shape) for ins in instructions if ins.has_weight)
        if weight_numel > 0:
            weights = weights.reshape(-1, weight_numel)
        del weight_numel

        # = extract individual input irreps =
        # If only one input irrep, can avoid creating a view
        if len(self.irreps_in1) == 1:
            x1_list = [x1s.reshape(batch_numel, self.irreps_in1[0].mul, self.irreps_in1[0].ir.dim)]
        else:
            x1_list = [
                x1s[:, i].reshape(batch_numel, mul_ir.mul, mul_ir.ir.dim) for i, mul_ir in zip(self.irreps_in1.slices(), self.irreps_in1)
            ]

        x2_list = []
        # If only one input irrep, can avoid creating a view
        if len(self.irreps_in2) == 1:
            x2_list.append(x2s.reshape(batch_numel, self.irreps_in2[0].mul, self.irreps_in2[0].ir.dim))
        else:
            for i, mul_ir in zip(self.irreps_in2.slices(), self.irreps_in2):
                x2_list.append(x2s[:, i].reshape(batch_numel, mul_ir.mul, mul_ir.ir.dim))

        # The einsum string index to prepend to the weights if the weights are not shared and have a batch dimension
        z = "" if self.shared_weights else "z"

        # Cache of input irrep pairs whose outer products (xx) have already been computed
        xx_dict = dict()

        # Current index in the flat weight tensor
        flat_weight_index = 0

        outputs = []

        for idx,ins in enumerate(instructions):
            mul_ir_in1 = self.irreps_in1[ins.i_in1]
            mul_ir_in2 = self.irreps_in2[ins.i_in2]
            mul_ir_out = self.irreps_out[ins.i_out]

            assert mul_ir_in1.ir.p * mul_ir_in2.ir.p == mul_ir_out.ir.p
            assert abs(mul_ir_in1.ir.l - mul_ir_in2.ir.l) <= mul_ir_out.ir.l <= mul_ir_in1.ir.l + mul_ir_in2.ir.l

            if mul_ir_in1.dim == 0 or mul_ir_in2.dim == 0 or mul_ir_out.dim == 0:
                continue

            x1 = x1_list[ins.i_in1]
            x2 = x2_list[ins.i_in2]

            assert ins.connection_mode in ["uvw", "uvu", "uvv", "uuw", "uuu", "uvuv", "uvu<v", "u<vw"]

            if ins.has_weight:
                # Extract the weight from the flattened weight tensor
                w = weights[:, flat_weight_index : flat_weight_index + prod(ins.path_shape)].reshape(
                    (() if self.shared_weights else (-1,)) + tuple(ins.path_shape)
                )
                flat_weight_index += prod(ins.path_shape)

            # Construct the general xx in case this instruction isn't specialized
            # If this isn't used, the dead code will get removed
            key = (ins.i_in1, ins.i_in2, ins.connection_mode[:2])
            if key not in xx_dict:
                if ins.connection_mode[:2] == "uu":
                    xx_dict[key] = torch.einsum("zui,zuj->zuij", x1, x2)
                else:
                    xx_dict[key] = torch.einsum("zui,zvj->zuvij", x1, x2)
            xx = xx_dict[key]
            del key

            # Create a proxy & request for the relevant wigner w3j
            # If not used (because of specialized code), will get removed later.
            # w3j_name = f"_w3j_{mul_ir_in1.ir.l}_{mul_ir_in2.ir.l}_{mul_ir_out.ir.l}"
            w3j = o3.wigner_3j(mul_ir_in1.ir.l, mul_ir_in2.ir.l, mul_ir_out.ir.l).to(x1s.device)

            l1l2l3 = (mul_ir_in1.ir.l, mul_ir_in2.ir.l, mul_ir_out.ir.l)

            if ins.connection_mode == "uvw":
                assert ins.has_weight
                if self._specialized_code and l1l2l3 == (0, 0, 0):
                    result = torch.einsum(
                        f"{z}uvw,zu,zv->zw", w, x1.reshape(batch_numel, mul_ir_in1.dim), x2.reshape(batch_numel, mul_ir_in2.dim)
                    )
                elif self._specialized_code and mul_ir_in1.ir.l == 0:
                    result = torch.einsum(f"{z}uvw,zu,zvj->zwj", w, x1.reshape(batch_numel, mul_ir_in1.dim), x2) / sqrt(
                        mul_ir_out.ir.dim
                    )
                elif self._specialized_code and mul_ir_in2.ir.l == 0:
                    result = torch.einsum(f"{z}uvw,zui,zv->zwi", w, x1, x2.reshape(batch_numel, mul_ir_in2.dim)) / sqrt(
                        mul_ir_out.ir.dim
                    )
                elif self._specialized_code and mul_ir_out.ir.l == 0:
                    result = torch.einsum(f"{z}uvw,zui,zvi->zw", w, x1, x2) / sqrt(mul_ir_in1.ir.dim)
                else:
                    result = torch.einsum(f"{z}uvw,ijk,zuvij->zwk", w, w3j, xx)
            if ins.connection_mode == "uvu":
                assert mul_ir_in1.mul == mul_ir_out.mul
                if ins.has_weight:
                    if self._specialized_code and l1l2l3 == (0, 0, 0):
                        result = torch.einsum(
                            f"{z}uv,zu,zv->zu", w, x1.reshape(batch_numel, mul_ir_in1.dim), x2.reshape(batch_numel, mul_ir_in2.dim)
                        )
                    elif self._specialized_code and mul_ir_in1.ir.l == 0:
                        result = torch.einsum(f"{z}uv,zu,zvj->zuj", w, x1.reshape(batch_numel, mul_ir_in1.dim), x2) / sqrt(
                            mul_ir_out.ir.dim
                        )
                    elif self._specialized_code and mul_ir_in2.ir.l == 0:
                        result = torch.einsum(f"{z}uv,zui,zv->zui", w, x1, x2.reshape(batch_numel, mul_ir_in2.dim)) / sqrt(
                            mul_ir_out.ir.dim
                        )
                    elif self._specialized_code and mul_ir_out.ir.l == 0:
                        result = torch.einsum(f"{z}uv,zui,zvi->zu", w, x1, x2) / sqrt(mul_ir_in1.ir.dim)
                    else:
                        result = torch.einsum(f"{z}uv,ijk,zuvij->zuk", w, w3j, xx)
                else:
                    # not so useful operation because v is summed
                    result = torch.einsum("ijk,zuvij->zuk", w3j, xx)
            if ins.connection_mode == "uvv":
                assert mul_ir_in2.mul == mul_ir_out.mul
                if ins.has_weight:
                    if self._specialized_code and l1l2l3 == (0, 0, 0):
                        result = torch.einsum(
                            f"{z}uv,zu,zv->zv", w, x1.reshape(batch_numel, mul_ir_in1.dim), x2.reshape(batch_numel, mul_ir_in2.dim)
                        )
                    elif self._specialized_code and mul_ir_in1.ir.l == 0:
                        result = torch.einsum(f"{z}uv,zu,zvj->zvj", w, x1.reshape(batch_numel, mul_ir_in1.dim), x2) / sqrt(
                            mul_ir_out.ir.dim
                        )
                    elif self._specialized_code and mul_ir_in2.ir.l == 0:
                        result = torch.einsum(f"{z}uv,zui,zv->zvi", w, x1, x2.reshape(batch_numel, mul_ir_in2.dim)) / sqrt(
                            mul_ir_out.ir.dim
                        )
                    elif self._specialized_code and mul_ir_out.ir.l == 0:
                        result = torch.einsum(f"{z}uv,zui,zvi->zv", w, x1, x2) / sqrt(mul_ir_in1.ir.dim)
                    else:
                        result = torch.einsum(f"{z}uv,ijk,zuvij->zvk", w, w3j, xx)
                else:
                    # not so useful operation because u is summed
                    # only specialize out for this path
                    if self._specialized_code and l1l2l3 == (0, 0, 0):
                        result = torch.einsum(
                            "zu,zv->zv", x1.reshape(batch_numel, mul_ir_in1.dim), x2.reshape(batch_numel, mul_ir_in2.dim)
                        )
                    elif self._specialized_code and mul_ir_in1.ir.l == 0:
                        result = torch.einsum("zu,zvj->zvj", x1.reshape(batch_numel, mul_ir_in1.dim), x2) / sqrt(mul_ir_out.ir.dim)
                    elif self._specialized_code and mul_ir_in2.ir.l == 0:
                        result = torch.einsum("zui,zv->zvi", x1, x2.reshape(batch_numel, mul_ir_in2.dim)) / sqrt(mul_ir_out.ir.dim)
                    elif self._specialized_code and mul_ir_out.ir.l == 0:
                        result = torch.einsum("zui,zvi->zv", x1, x2) / sqrt(mul_ir_in1.ir.dim)
                    else:
                        result = torch.einsum("ijk,zuvij->zvk", w3j, xx)
            if ins.connection_mode == "uuw":
                assert mul_ir_in1.mul == mul_ir_in2.mul
                if ins.has_weight:
                    if self._specialized_code and l1l2l3 == (0, 0, 0):
                        result = torch.einsum(
                            f"{z}uw,zu,zu->zw", w, x1.reshape(batch_numel, mul_ir_in1.dim), x2.reshape(batch_numel, mul_ir_in2.dim)
                        )
                    elif self._specialized_code and mul_ir_in1.ir.l == 0:
                        result = torch.einsum(f"{z}uw,zu,zuj->zwj", w, x1.reshape(batch_numel, mul_ir_in1.dim), x2) / sqrt(
                            mul_ir_out.ir.dim
                        )
                    elif self._specialized_code and mul_ir_in2.ir.l == 0:
                        result = torch.einsum(f"{z}uw,zui,zu->zwi", w, x1, x2.reshape(batch_numel, mul_ir_in2.dim)) / sqrt(
                            mul_ir_out.ir.dim
                        )
                    elif self._specialized_code and mul_ir_out.ir.l == 0:
                        result = torch.einsum(f"{z}uw,zui,zui->zw", w, x1, x2) / sqrt(mul_ir_in1.ir.dim)
                    else:
                        result = torch.einsum(f"{z}uw,ijk,zuij->zwk", w, w3j, xx)
                else:
                    # equivalent to tp(x, y, 'uuu').sum('u')
                    assert mul_ir_out.mul == 1
                    result = torch.einsum("ijk,zuij->zk", w3j, xx)
            if ins.connection_mode == "uuu":
                assert mul_ir_in1.mul == mul_ir_in2.mul == mul_ir_out.mul
                if ins.has_weight:
                    if self._specialized_code and l1l2l3 == (0, 0, 0):
                        result = torch.einsum(
                            f"{z}u,zu,zu->zu", w, x1.reshape(batch_numel, mul_ir_in1.dim), x2.reshape(batch_numel, mul_ir_in2.dim)
                        )
                    elif self._specialized_code and l1l2l3 == (1, 1, 1):
                        result = torch.einsum(f"{z}u,zui->zui", w, torch.cross(x1, x2, dim=2)) / sqrt(2 * 3)
                    elif self._specialized_code and mul_ir_in1.ir.l == 0:
                        result = torch.einsum(f"{z}u,zu,zuj->zuj", w, x1.reshape(batch_numel, mul_ir_in1.dim), x2) / sqrt(
                            mul_ir_out.ir.dim
                        )
                    elif self._specialized_code and mul_ir_in2.ir.l == 0:
                        result = torch.einsum(f"{z}u,zui,zu->zui", w, x1, x2.reshape(batch_numel, mul_ir_in2.dim)) / sqrt(
                            mul_ir_out.ir.dim
                        )
                    elif self._specialized_code and mul_ir_out.ir.l == 0:
                        result = torch.einsum(f"{z}u,zui,zui->zu", w, x1, x2) / sqrt(mul_ir_in1.ir.dim)
                    else:
                        result = torch.einsum(f"{z}u,ijk,zuij->zuk", w, w3j, xx)
                else:
                    if self._specialized_code and l1l2l3 == (0, 0, 0):
                        result = torch.einsum(
                            "zu,zu->zu", x1.reshape(batch_numel, mul_ir_in1.dim), x2.reshape(batch_numel, mul_ir_in2.dim)
                        )
                    elif self._specialized_code and l1l2l3 == (1, 1, 1):
                        result = torch.cross(x1, x2, dim=2) * (1.0 / sqrt(2 * 3))
                    elif self._specialized_code and mul_ir_in1.ir.l == 0:
                        result = torch.einsum("zu,zuj->zuj", x1.reshape(batch_numel, mul_ir_in1.dim), x2) / sqrt(mul_ir_out.ir.dim)
                    elif self._specialized_code and mul_ir_in2.ir.l == 0:
                        result = torch.einsum("zui,zu->zui", x1, x2.reshape(batch_numel, mul_ir_in2.dim)) / sqrt(mul_ir_out.ir.dim)
                    elif self._specialized_code and mul_ir_out.ir.l == 0:
                        result = torch.einsum("zui,zui->zu", x1, x2) / sqrt(mul_ir_in1.ir.dim)
                    else:
                        result = torch.einsum("ijk,zuij->zuk", w3j, xx)
            if ins.connection_mode == "uvuv":
                assert mul_ir_in1.mul * mul_ir_in2.mul == mul_ir_out.mul
                if ins.has_weight:
                    # TODO implement specialized code
                    result = torch.einsum(f"{z}uv,ijk,zuvij->zuvk", w, w3j, xx)
                else:
                    # TODO implement specialized code
                    result = torch.einsum("ijk,zuvij->zuvk", w3j, xx)
            if ins.connection_mode == "uvu<v":
                assert mul_ir_in1.mul == mul_ir_in2.mul
                assert mul_ir_in1.mul * (mul_ir_in1.mul - 1) // 2 == mul_ir_out.mul
                # name = f"_triu_indices_{mul_ir_in1.mul}"
                # constants[name] = torch.triu_indices(mul_ir_in1.mul, mul_ir_in1.mul, 1)
                # i = fx.Proxy(graph.get_attr(name), tracer=tracer)
                i = torch.triu_indices(mul_ir_in1.mul, mul_ir_in1.mul, 1)
                xx = xx[:, i[0], i[1]]  # zuvij -> zwij
                if ins.has_weight:
                    # TODO implement specialized code
                    result = torch.einsum(f"{z}w,ijk,zwij->zwk", w, w3j, xx)
                else:
                    # TODO implement specialized code
                    result = torch.einsum("ijk,zwij->zwk", w3j, xx)
            if ins.connection_mode == "u<vw":
                assert mul_ir_in1.mul == mul_ir_in2.mul
                assert ins.has_weight
                # name = f"_triu_indices_{mul_ir_in1.mul}"
                # constants[name] = torch.triu_indices(mul_ir_in1.mul, mul_ir_in1.mul, 1)
                # i = fx.Proxy(graph.get_attr(name), tracer=tracer)
                i = torch.triu_indices(mul_ir_in1.mul, mul_ir_in1.mul, 1)
                xx = xx[:, i[0], i[1]]  # zuvij -> zqij
                # TODO implement specialized code
                result = torch.einsum(f"{z}qw,ijk,zqij->zwk", w, w3j, xx)

            result = ins.path_weight * result

            outputs += [result.reshape(batch_numel, mul_ir_out.dim)]


        # = Return the result =
        outputs = [
            self._sum_tensors(
                [out for ins, out in zip(instructions, outputs) if ins.i_out == i_out],
                shape=(batch_numel, mul_ir_out.dim),
                like=x1s,
            )
            for i_out, mul_ir_out in enumerate(self.irreps_out)
            if mul_ir_out.mul > 0
        ]
        if len(outputs) > 1:
            outputs = torch.cat(outputs, dim=1)
        else:
            # Avoid an unnecessary copy in a size one torch.cat
            outputs = outputs[0]

        outputs = outputs.reshape(output_shape)

        return outputs

    def pick_path(self, path_weight, sparsity): 
        # 计算总共需要选择的元素数量（30%的tensor大小）  
        num_elements = path_weight.numel()  
        num_elements_to_select = int(sparsity * num_elements)+1  

        # selected_indices = torch.multinomial(path_weight, num_elements_to_select, replacement=False)

        _, selected_indices = torch.topk(path_weight, num_elements_to_select)  

        return selected_indices
        
        # # 计算正排选择的元素数量（30% - er）  
        # num_top_elements = num_elements_to_select - int(self.er * num_elements)  
        
        # # 计算随机选择的元素数量（er比例的数量）  
        # num_random_elements = num_elements_to_select - num_top_elements  
        
        # # 获取top (30% - er) 比例的元素的indices  
        # _, top_indices = torch.topk(path_weight, num_top_elements)  
        
        # # 获取剩余元素的indices  
        # remaining_indices = torch.tensor([i for i in range(num_elements) if i not in top_indices])  
        
        # # 从剩余元素中不放回地随机选择er比例的元素  
        # random_indices = remaining_indices[torch.randperm(len(remaining_indices))[:num_random_elements]].to(path_weight.device)
        
        # # 合并两部分的indices  
        # final_indices = torch.cat((top_indices, random_indices)).to(torch.int64) 

        # return final_indices

    def forward(self, x, y, weight: Optional[torch.Tensor] = None):
        r"""Evaluate :math:`w x \otimes y`.

        Parameters
        ----------
        x : `torch.Tensor`
            tensor of shape ``(..., irreps_in1.dim)``

        y : `torch.Tensor`
            tensor of shape ``(..., irreps_in2.dim)``

        weight : `torch.Tensor` or list of `torch.Tensor`, optional
            required if ``internal_weights`` is ``False``
            tensor of shape ``(self.weight_numel,)`` if ``shared_weights`` is ``True``
            tensor of shape ``(..., self.weight_numel)`` if ``shared_weights`` is ``False``
            or list of tensors of shapes ``weight_shape`` / ``(...) + weight_shape``.
            Use ``self.instructions`` to know what are the weights used for.

        Returns
        -------
        `torch.Tensor`
            tensor of shape ``(..., irreps_out.dim)``
        """
        assert x.shape[-1] == self._in1_dim, "Incorrect last dimension for x"
        assert y.shape[-1] == self._in2_dim, "Incorrect last dimension for y"

        # - PROFILER - with torch.autograd.profiler.record_function(self._profiling_str):
        real_weight = self._get_weights(weight)
        # return self._handwrite_main_left_right(x,y,real_weight, self.instructions)
        
        # path_mask = torch.ones(len(self.instructions))# if path_mask is None else path_mask
        # return self._compiled_main_left_right(x, y, real_weight, path_mask)

        instruction_weight = torch.tensor([ins.path_weight for ins in self.instructions]).to(x.device)

        path_input = torch.concat((x[:, self.irreps_in1.slices()[0]],y[:, self.irreps_in2.slices()[0]]),dim=-1)
        path_weight = self.path_layer(path_input).sum(axis=0)   # learnable version

        if self.tp_times < 20000:
            min_val = torch.min(path_weight)  
            range_val = torch.max(path_weight) - min_val  
            scaled_tensor = (path_weight - min_val) / range_val  
            path_weight = nn.Softmax(dim=0)(scaled_tensor)
            path_mask = self.pick_path(path_weight,sparsity=1-self.sparsity) # pick 30%
            if self.tp_times % 200 == 0:
                self.sparsity += 0.007
        else:
            _, path_mask = torch.topk(path_weight, int(path_weight.shape[0]*(1-0.7))+1) # pick top 30%
        
        self.tp_times += 1

        instructions = []
        indice = []
        count = 0
        path_weight_sum = instruction_weight[path_mask].sum()
        for i in range(len(self.instructions)):
            current_path = prod(self.instructions[i].path_shape)
            if i in path_mask:
                instructions.append(self.instructions[i])
                instructions[-1]._replace(path_weight=instructions[-1].path_weight/path_weight_sum)
                indice+=list(range(count,count+current_path))
            count += current_path

        if real_weight.numel() != 0:
            real_weight = real_weight[indice] if len(real_weight.shape)==1 else real_weight[:,indice]
        return self._handwrite_main_left_right(x,y,real_weight, instructions)

    def weight_view_for_instruction(self, instruction: int, weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        r"""View of weights corresponding to ``instruction``.

        Parameters
        ----------
        instruction : int
            The index of the instruction to get a view on the weights for. ``self.instructions[instruction].has_weight`` must
            be ``True``.

        weight : `torch.Tensor`, optional
            like ``weight`` argument to ``forward()``

        Returns
        -------
        `torch.Tensor`
            A view on ``weight`` or this object's internal weights for the weights corresponding to the ``instruction`` th
            instruction.
        """
        if not self.instructions[instruction].has_weight:
            raise ValueError(f"Instruction {instruction} has no weights.")
        offset = sum(prod(ins.path_shape) for ins in self.instructions[:instruction])
        ins = self.instructions[instruction]
        weight = self._get_weights(weight)
        batchshape = weight.shape[:-1]
        return weight.narrow(-1, offset, prod(ins.path_shape)).view(batchshape + ins.path_shape)

    def weight_views(self, weight: Optional[torch.Tensor] = None, yield_instruction: bool = False):
        r"""Iterator over weight views for each weighted instruction.

        Parameters
        ----------
        weight : `torch.Tensor`, optional
            like ``weight`` argument to ``forward()``

        yield_instruction : `bool`, default False
            Whether to also yield the corresponding instruction.

        Yields
        ------
        If ``yield_instruction`` is ``True``, yields ``(instruction_index, instruction, weight_view)``.
        Otherwise, yields ``weight_view``.
        """
        weight = self._get_weights(weight)
        batchshape = weight.shape[:-1]
        offset = 0
        for ins_i, ins in enumerate(self.instructions):
            if ins.has_weight:
                flatsize = prod(ins.path_shape)
                this_weight = weight.narrow(-1, offset, flatsize).view(batchshape + ins.path_shape)
                offset += flatsize
                if yield_instruction:
                    yield ins_i, ins, this_weight
                else:
                    yield this_weight

    def visualize(
        self, weight: Optional[torch.Tensor] = None, plot_weight: bool = True, aspect_ratio=1, ax=None
    ):  # pragma: no cover
        r"""Visualize the connectivity of this `e3nn.o3.TensorProduct`

        Parameters
        ----------
        weight : `torch.Tensor`, optional
            like ``weight`` argument to ``forward()``

        plot_weight : `bool`, default True
            Whether to color paths by the sum of their weights.

        ax : ``matplotlib.Axes``, default None
            The axes to plot on. If ``None``, a new figure will be created.

        Returns
        -------
        (fig, ax)
            The figure and axes on which the plot was drawn.
        """
        import numpy as np

        def _intersection(x, u, y, v):
            u2 = np.sum(u**2)
            v2 = np.sum(v**2)
            uv = np.sum(u * v)
            det = u2 * v2 - uv**2
            mu = np.sum((u * uv - v * u2) * (y - x)) / det
            return y + mu * v

        import matplotlib
        import matplotlib.pyplot as plt
        from matplotlib import patches
        from matplotlib.path import Path

        if ax is None:
            ax = plt.gca()

        fig = ax.get_figure()

        # hexagon
        verts = [np.array([np.cos(a * 2 * np.pi / 6), np.sin(a * 2 * np.pi / 6)]) for a in range(6)]
        verts = np.asarray(verts)

        # scale it
        if not (aspect_ratio in ["auto"] or isinstance(aspect_ratio, (float, int))):
            raise ValueError(f"aspect_ratio must be 'auto' or a float or int, got {aspect_ratio}")

        if aspect_ratio == "auto":
            factor = 0.2 / 2
            min_aspect = 1 / 2
            h_factor = max(len(self.irreps_in2), len(self.irreps_in1))
            w_factor = len(self.irreps_out)
            if h_factor / w_factor < min_aspect:
                h_factor = min_aspect * w_factor
            verts[:, 1] *= h_factor * factor
            verts[:, 0] *= w_factor * factor

        if isinstance(aspect_ratio, (float, int)):
            factor = 0.1 * max(len(self.irreps_in2), len(self.irreps_in1), len(self.irreps_out))
            verts[:, 1] *= factor
            verts[:, 0] *= aspect_ratio * factor

        codes = [
            Path.MOVETO,
            Path.LINETO,
            Path.MOVETO,
            Path.LINETO,
            Path.MOVETO,
            Path.LINETO,
        ]

        path = Path(verts, codes)
        patch = patches.PathPatch(path, facecolor="none", lw=1, zorder=2)
        ax.add_patch(patch)

        n = len(self.irreps_in1)
        b, a = verts[2:4]

        c_in1 = (a + b) / 2
        s_in1 = [a + (i + 1) / (n + 1) * (b - a) for i in range(n)]

        n = len(self.irreps_in2)
        b, a = verts[:2]

        c_in2 = (a + b) / 2
        s_in2 = [a + (i + 1) / (n + 1) * (b - a) for i in range(n)]

        n = len(self.irreps_out)
        a, b = verts[4:6]

        s_out = [a + (i + 1) / (n + 1) * (b - a) for i in range(n)]

        # get weights
        if weight is None and not self.internal_weights:
            plot_weight = False
        elif plot_weight:
            with torch.no_grad():
                path_weight = []
                for ins_i, ins in enumerate(self.instructions):
                    if ins.has_weight:
                        this_weight = self.weight_view_for_instruction(ins_i, weight=weight).cpu()
                        path_weight.append(this_weight.pow(2).mean())
                    else:
                        path_weight.append(0)
                path_weight = np.asarray(path_weight)
                path_weight /= np.abs(path_weight).max()
        cmap = matplotlib.cm.get_cmap("Blues")

        for ins_index, ins in enumerate(self.instructions):
            y = _intersection(s_in1[ins.i_in1], c_in1, s_in2[ins.i_in2], c_in2)

            verts = []
            codes = []
            verts += [s_out[ins.i_out], y]
            codes += [Path.MOVETO, Path.LINETO]
            verts += [s_in1[ins.i_in1], y]
            codes += [Path.MOVETO, Path.LINETO]
            verts += [s_in2[ins.i_in2], y]
            codes += [Path.MOVETO, Path.LINETO]

            if plot_weight:
                color = cmap(0.5 + 0.5 * path_weight[ins_index]) if ins.has_weight else "black"
            else:
                color = "green" if ins.has_weight else "black"

            ax.add_patch(
                patches.PathPatch(
                    Path(verts, codes),
                    facecolor="none",
                    edgecolor=color,
                    alpha=0.5,
                    ls="-",
                    lw=1.5,
                )
            )

        # add labels
        padding = 3
        fontsize = 10

        def format_ir(mul_ir):
            if mul_ir.mul == 1:
                return f"${mul_ir.ir}$"
            return f"${mul_ir.mul} \\times {mul_ir.ir}$"

        for i, mul_ir in enumerate(self.irreps_in1):
            ax.annotate(
                format_ir(mul_ir),
                s_in1[i],
                horizontalalignment="right",
                textcoords="offset points",
                xytext=(-padding, 0),
                fontsize=fontsize,
            )

        for i, mul_ir in enumerate(self.irreps_in2):
            ax.annotate(
                format_ir(mul_ir),
                s_in2[i],
                horizontalalignment="left",
                textcoords="offset points",
                xytext=(padding, 0),
                fontsize=fontsize,
            )

        for i, mul_ir in enumerate(self.irreps_out):
            ax.annotate(
                format_ir(mul_ir),
                s_out[i],
                horizontalalignment="center",
                verticalalignment="top",
                rotation=90,
                textcoords="offset points",
                xytext=(0, -padding),
                fontsize=fontsize,
            )

        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.axis("equal")
        ax.axis("off")

        return fig, ax


def construct_o3irrps(dim,order):
    string = []
    for l in range(order+1):
        string.append(f"{dim}x{l}e" if l%2==0 else f"{dim}x{l}o")
    return "+".join(string)

class SO2_m_Convolution_general(torch.nn.Module):
    """
    SO(2) Conv: Perform an SO(2) convolution on features corresponding to +- m

    Args:
        m (int):                    Order of the spherical harmonic coefficients
        sphere_channels (int):      Number of spherical channels
        m_output_channels (int):    Number of output channels used during the SO(2) conv
        lmax_list (list:int):       List of degrees (l) for each resolution
        mmax_list (list:int):       List of orders (m) for each resolution
    """
    def __init__(
        self,
        m, 
        sphere_channels,
        m_output_channels,
        lmax_list, 
        mmax_list
    ):
        super(SO2_m_Convolution_general, self).__init__()
        
        self.m = m
        self.sphere_channels = sphere_channels
        self.m_output_channels = m_output_channels
        self.lmax_list = lmax_list
        self.mmax_list = mmax_list
        self.num_resolutions = len(self.lmax_list)

        num_channels = 0
        for i in range(self.num_resolutions):
            num_coefficents = 0
            if self.mmax_list[i] >= self.m:
                num_coefficents = self.lmax_list[i] - self.m + 1
            num_channels = num_channels + num_coefficents * self.sphere_channels
        assert num_channels > 0

        self.fc = Linear(num_channels, 
            2 * self.m_output_channels * (num_channels // self.sphere_channels), 
            bias=False)
        self.fc.weight.data.mul_(1 / math.sqrt(2))


    def forward(self, x_m):
        x_m = self.fc(x_m)
        x_r = x_m.narrow(2, 0, self.fc.out_features // 2)
        x_i = x_m.narrow(2, self.fc.out_features // 2, self.fc.out_features // 2)
        x_m_r = x_r.narrow(1, 0, 1) - x_i.narrow(1, 1, 1) #x_r[:, 0] - x_i[:, 1]
        x_m_i = x_r.narrow(1, 1, 1) + x_i.narrow(1, 0, 1) #x_r[:, 1] + x_i[:, 0]
        x_out = torch.cat((x_m_r, x_m_i), dim=1)
        
        return x_out

class SO2_Convolution_general(torch.nn.Module):
    """
    SO(2) Block: Perform SO(2) convolutions for all m (orders)

    Args:
        sphere_channels (int):      Number of spherical channels
        m_output_channels (int):    Number of output channels used during the SO(2) conv
        lmax_list (list:int):       List of degrees (l) for each resolution
        mmax_list (list:int):       List of orders (m) for each resolution
        mappingReduced (CoefficientMappingModule): Used to extract a subset of m components
        internal_weights (bool):    If True, not using radial function to multiply inputs features
        edge_channels_list (list:int):  List of sizes of invariant edge embedding. For example, [input_channels, hidden_channels, hidden_channels].
        extra_m0_output_channels (int): If not None, return `out_embedding` (SO3_Embedding) and `extra_m0_features` (Tensor).
    """
    def __init__(
        self,
        sphere_channels,
        m_output_channels,
        lmax_list,
        mmax_list,
        mappingReduced,
        internal_weights=True,
        edge_channels_list=None,
        extra_m0_output_channels=None
    ):
        super(SO2_Convolution_general, self).__init__()
        self.sphere_channels = sphere_channels
        self.m_output_channels = m_output_channels
        self.lmax_list = lmax_list
        self.mmax_list = mmax_list
        self.mappingReduced = mappingReduced
        self.num_resolutions = len(lmax_list)
        self.internal_weights = internal_weights
        self.edge_channels_list = copy.deepcopy(edge_channels_list)
        self.extra_m0_output_channels = extra_m0_output_channels

        num_channels_rad = 0    # for radial function

        num_channels_m0 = 0
        for i in range(self.num_resolutions):
            num_coefficients = self.lmax_list[i] + 1
            num_channels_m0 = num_channels_m0 + num_coefficients * self.sphere_channels

        # SO(2) convolution for m = 0
        m0_output_channels = self.m_output_channels * (num_channels_m0 // self.sphere_channels)
        if self.extra_m0_output_channels is not None:
            m0_output_channels = m0_output_channels + self.extra_m0_output_channels
        self.fc_m0 = Linear(num_channels_m0, m0_output_channels)
        num_channels_rad = num_channels_rad + self.fc_m0.in_features
        
        # SO(2) convolution for non-zero m
        self.so2_m_conv = nn.ModuleList()
        for m in range(1, max(self.mmax_list) + 1):
            self.so2_m_conv.append(
                SO2_m_Convolution_general(
                    m, 
                    self.sphere_channels,
                    self.m_output_channels,
                    self.lmax_list, 
                    self.mmax_list,
                )
            )
            num_channels_rad = num_channels_rad + self.so2_m_conv[-1].fc.in_features

        # Embedding function of distance
        self.rad_func = None
        if not self.internal_weights:
            assert self.edge_channels_list is not None
            self.edge_channels_list.append(int(num_channels_rad))
            self.rad_func = RadialFunction(self.edge_channels_list)


    def forward(self, x, x_edge):

        num_edges = len(x_edge)
        out = []

        # Reshape the spherical harmonics based on m (order)
        x._m_primary(self.mappingReduced)

        # radial function
        if self.rad_func is not None:
            x_edge = self.rad_func(x_edge)
        offset_rad = 0

        # Compute m=0 coefficients separately since they only have real values (no imaginary)
        x_0 = x.embedding.narrow(1, 0, self.mappingReduced.m_size[0])
        x_0 = x_0.reshape(num_edges, -1)
        if self.rad_func is not None:
            x_edge_0 = x_edge.narrow(1, 0, self.fc_m0.in_features)
            x_0 = x_0 * x_edge_0
        x_0 = self.fc_m0(x_0)

        x_0_extra = None
        # extract extra m0 features 
        if self.extra_m0_output_channels is not None:
            x_0_extra = x_0.narrow(-1, 0, self.extra_m0_output_channels)
            x_0 = x_0.narrow(-1, self.extra_m0_output_channels, (self.fc_m0.out_features - self.extra_m0_output_channels))
        
        x_0 = x_0.view(num_edges, -1, self.m_output_channels)
        #x.embedding[:, 0 : self.mappingReduced.m_size[0]] = x_0
        out.append(x_0)
        offset_rad = offset_rad + self.fc_m0.in_features

        # Compute the values for the m > 0 coefficients
        offset = self.mappingReduced.m_size[0]
        for m in range(1, max(self.mmax_list) + 1):
            # Get the m order coefficients
            x_m = x.embedding.narrow(1, offset, 2 * self.mappingReduced.m_size[m])
            x_m = x_m.reshape(num_edges, 2, -1)

            # Perform SO(2) convolution
            if self.rad_func is not None:
                x_edge_m = x_edge.narrow(1, offset_rad, self.so2_m_conv[m - 1].fc.in_features)
                x_edge_m = x_edge_m.reshape(num_edges, 1, self.so2_m_conv[m - 1].fc.in_features)
                x_m = x_m * x_edge_m
            x_m = self.so2_m_conv[m - 1](x_m)
            x_m = x_m.view(num_edges, -1, self.m_output_channels)
            #x.embedding[:, offset : offset + 2 * self.mappingReduced.m_size[m]] = x_m
            out.append(x_m)
            offset = offset + 2 * self.mappingReduced.m_size[m]
            offset_rad = offset_rad + self.so2_m_conv[m - 1].fc.in_features

        out = torch.cat(out, dim=1)
        out_embedding = SO3_Embedding(
            0, 
            x.lmax_list.copy(), 
            self.m_output_channels, 
            device=x.device, 
            dtype=x.dtype
        )
        out_embedding.set_embedding(out)
        out_embedding.set_lmax_mmax(self.lmax_list.copy(), self.mmax_list.copy())

        # Reshape the spherical harmonics based on l (degree)
        out_embedding._l_primary(self.mappingReduced)

        if self.extra_m0_output_channels is not None:
            return out_embedding, x_0_extra
        else:
            return out_embedding


class SO2Tensorproduct(torch.nn.Module):
    """
    SO2EquivariantGraphAttention: Perform MLP attention + non-linear message passing
        SO(2) Convolution with radial function -> S2 Activation -> SO(2) Convolution -> attention weights and non-linear messages
        attention weights * non-linear messages -> Linear

    Args:
        sphere_channels (int):      Number of spherical channels
        hidden_channels (int):      Number of hidden channels used during the SO(2) conv
        num_heads (int):            Number of attention heads
        attn_alpha_head (int):      Number of channels for alpha vector in each attention head
        attn_value_head (int):      Number of channels for value vector in each attention head
        output_channels (int):      Number of output channels
        lmax_list (list:int):       List of degrees (l) for each resolution
        mmax_list (list:int):       List of orders (m) for each resolution
        
        SO3_rotation (list:SO3_Rotation): Class to calculate Wigner-D matrices and rotate embeddings
        mappingReduced (CoefficientMappingModule): Class to convert l and m indices once node embedding is rotated
        SO3_grid (SO3_grid):        Class used to convert from grid the spherical harmonic representations

        max_num_elements (int):     Maximum number of atomic numbers
        edge_channels_list (list:int):  List of sizes of invariant edge embedding. For example, [input_channels, hidden_channels, hidden_channels].
                                        The last one will be used as hidden size when `use_atom_edge_embedding` is `True`.
        use_atom_edge_embedding (bool): Whether to use atomic embedding along with relative distance for edge scalar features
        use_m_share_rad (bool):     Whether all m components within a type-L vector of one channel share radial function weights

        activation (str):           Type of activation function
        use_s2_act_attn (bool):     Whether to use attention after S2 activation. Otherwise, use the same attention as Equiformer
        use_attn_renorm (bool):     Whether to re-normalize attention weights
        use_gate_act (bool):        If `True`, use gate activation. Otherwise, use S2 activation.
        use_sep_s2_act (bool):      If `True`, use separable S2 activation when `use_gate_act` is False.

        alpha_drop (float):         Dropout rate for attention weights
    """

    def __init__(
        self,
        irrep_in,
        sphere_channels,
        hidden_channels,
        num_heads, 
        attn_alpha_channels,
        attn_value_channels, 
        output_channels,
        lmax_list,
        mmax_list,
        mappingReduced, 
        SO3_grid, 
        max_num_elements,
        edge_channels_list,
        use_atom_edge_embedding=True, 
        use_m_share_rad=False,
        activation='scaled_silu', 
        use_s2_act_attn=False, 
        use_attn_renorm=True,
        use_gate_act=False, 
        use_sep_s2_act=True,
        alpha_drop=0.0,
        so2_dim=16,
    ):
        super(SO2Tensorproduct, self).__init__()
        
        self.so2_dim = so2_dim
        self.irrep_in = irrep_in
        self.order = len(self.irrep_in)-1
        self.sphere_channels = sphere_channels
        self.hidden_channels = hidden_channels
        self.num_heads = num_heads
        self.attn_alpha_channels = attn_alpha_channels
        self.attn_value_channels = attn_value_channels
        self.output_channels = output_channels
        self.lmax_list = lmax_list
        self.mmax_list = mmax_list
        self.num_resolutions = len(self.lmax_list)
        
        # self.SO3_rotation = SO3_rotation
        self.SO3_rotation = nn.ModuleList()
        for i in range(self.so2_dim):
            self.SO3_rotation.append(SO3_Rotation(self.lmax_list[0]))

        self.mappingReduced = mappingReduced
        self.SO3_grid = SO3_grid
        
        # Create edge scalar (invariant to rotations) features
        # Embedding function of the atomic numbers
        self.max_num_elements = max_num_elements
        self.edge_channels_list = copy.deepcopy(edge_channels_list)
        self.use_atom_edge_embedding = use_atom_edge_embedding
        self.use_m_share_rad = use_m_share_rad

        if self.use_atom_edge_embedding:
            self.source_embedding = nn.Embedding(self.max_num_elements, self.edge_channels_list[-1])
            self.target_embedding = nn.Embedding(self.max_num_elements, self.edge_channels_list[-1])
            nn.init.uniform_(self.source_embedding.weight.data, -0.001, 0.001)
            nn.init.uniform_(self.target_embedding.weight.data, -0.001, 0.001)
            self.edge_channels_list[0] = self.edge_channels_list[0] + 2 * self.edge_channels_list[-1]
        else:
            self.source_embedding, self.target_embedding = None, None
        
        self.use_s2_act_attn    = use_s2_act_attn
        self.use_attn_renorm    = use_attn_renorm
        self.use_gate_act       = use_gate_act
        self.use_sep_s2_act     = use_sep_s2_act
        
        assert not self.use_s2_act_attn     # since this is not used
        
        # Create SO(2) convolution blocks
        extra_m0_output_channels = None
        if not self.use_s2_act_attn:
            extra_m0_output_channels = self.num_heads * self.attn_alpha_channels
            if self.use_gate_act:
                extra_m0_output_channels = extra_m0_output_channels + max(self.lmax_list) * self.hidden_channels
            else:
                if self.use_sep_s2_act:
                    extra_m0_output_channels = extra_m0_output_channels + self.hidden_channels
        
        if self.use_m_share_rad:
            self.edge_channels_list = self.edge_channels_list + [2 * self.sphere_channels * (max(self.lmax_list) + 1)]
            self.rad_func = RadialFunction(self.edge_channels_list)
            expand_index = torch.zeros([(max(self.lmax_list) + 1) ** 2]).long()
            for l in range(max(self.lmax_list) + 1):
                start_idx = l ** 2
                length = 2 * l + 1
                expand_index[start_idx : (start_idx + length)] = l
            self.register_buffer('expand_index', expand_index)

        self.so2_conv_1 = SO2_Convolution_general(
            2 * self.sphere_channels,
            self.hidden_channels,
            self.lmax_list,
            self.mmax_list,
            self.mappingReduced,
            internal_weights=(
                False if not self.use_m_share_rad 
                else True
            ),
            edge_channels_list=(
                self.edge_channels_list if not self.use_m_share_rad 
                else None
            ), 
            extra_m0_output_channels=extra_m0_output_channels # for attention weights and/or gate activation
        )      
        self.linear_down = Linear(            
            irreps_in=self.irrep_in,
            irreps_out=o3.Irreps(construct_o3irrps(self.so2_dim, order=self.order)),
            internal_weights=True,
            shared_weights=True,
            biases=True)

    def _init_edge_rot_mat(self, v, target):  
        """  
        生成一个Householder变换矩阵,将v旋转到target。  
        """  
        # 确保v和target是单位向量  
        v = v / torch.norm(v)  
        target = target / torch.norm(target)  
        
        # 计算差向量w并归一化得到v  
        w = v - target  
        u = w / torch.norm(w)  
        
        # 构造Householder矩阵H  
        I = torch.eye(len(v))  
        H = I - 2.0 * torch.outer(u, u)  
        
        return H
    

        
    def forward(
        self,
        x,
        y,
        atomic_numbers,
        edge_distance,
        edge_index
    ):

        y = self.linear_down(y)
        batch_size = y.shape[0]

        # Compute rotation matrix per order
        for L in range(1, self.order):
            target = torch.zeros((2*L+1)).to(y.device)
            target[L+1] = 1
            edge_rot_mat = self._init_edge_rot_mat(
                y[:, L**2*self.so2_dim:(L+1)**2*self.so2_dim].reshape(batch_size, self.so2_dim, -1), target    # bs,16,nxn
            )

            # Initialize the WignerD matrices and other values for spherical harmonic calculations
            for i in range(self.so2_dim):
                self.SO3_rotation[i].set_wigner(edge_rot_mat[:,i])



        x_message = SO3_Embedding(
            0,
            x.lmax_list.copy(), 
            x.num_channels * 2, 
            device=x.device, 
            dtype=x.dtype
        )
        x_message.set_embedding(x)
        x_message.set_lmax_mmax(self.lmax_list.copy(), self.mmax_list.copy())

        # Rotate the irreps to align with the edge
        x_message._rotate(self.SO3_rotation, self.lmax_list, self.mmax_list)

        # First SO(2)-convolution
        if self.use_s2_act_attn:
            x_message = self.so2_conv_1(x_message, y)
        else:
            x_message, x_0_extra = self.so2_conv_1(x_message, y)

        # Rotate back the irreps
        x_message._rotate_inv(self.SO3_rotation, self.mappingReduced)

        # Compute the sum of the incoming neighboring messages for each target node
        x_message._reduce_edge(edge_index[1], len(x.embedding))

        return x_message



// SPDX-License-Identifier: MIT
// Part of psfi — https://github.com/NickAlger/psf_interpolation
//
// Python bindings. Array-layout convention at the Python boundary: POINTS ARE
// ROWS, matching numpy/scipy practice (and the etree bindings) — point sets
// are (n, d), mesh vertices (num_vertices, d), mesh cells (num_cells, d+1),
// covariance stacks (n, d, d). Internally psfi stores points as columns; the
// transpose happens here, once, at the boundary.

#include <optional>
#include <tuple>
#include <utility>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>

#include "psfi/psfi.hpp"

namespace py = pybind11;
using namespace pybind11::literals;
using namespace psfi;

namespace {

using RowsXd = Eigen::Ref<const Eigen::MatrixXd>;
using RowsXi = Eigen::Ref<const Eigen::MatrixXi>;

Eigen::MatrixXd cols_from_rows( const RowsXd& rows )  { return rows.transpose(); }
Eigen::MatrixXi icols_from_rows( const RowsXi& rows ) { return rows.transpose(); }

// (n, d, d) numpy stack -> one matrix per entry.
std::vector<Eigen::MatrixXd> matrices_from_stack( const py::array_t<double>& stack, const char* name )
{
    if ( stack.ndim() != 3 || stack.shape(1) != stack.shape(2) )
    {
        throw std::invalid_argument(std::string(name) + " must have shape (n, d, d)");
    }
    auto A = stack.unchecked<3>();
    const int n = static_cast<int>(stack.shape(0));
    const int d = static_cast<int>(stack.shape(1));
    std::vector<Eigen::MatrixXd> out(n);
    for ( int ii = 0; ii < n; ++ii )
    {
        Eigen::MatrixXd M(d, d);
        for ( int rr = 0; rr < d; ++rr )
        {
            for ( int cc = 0; cc < d; ++cc )
            {
                M(rr, cc) = A(ii, rr, cc);
            }
        }
        out[ii] = std::move(M);
    }
    return out;
}

// (n, d, d) numpy stack -> (d*d, n) matrix with column v = vec(Sigma_v)
// (column-major vec, matching ImpulseResponseField::set_moment_fields).
Eigen::MatrixXd flat_field_from_stack( const py::array_t<double>& stack, const char* name )
{
    if ( stack.ndim() != 3 || stack.shape(1) != stack.shape(2) )
    {
        throw std::invalid_argument(std::string(name) + " must have shape (num_vertices, d, d)");
    }
    auto A = stack.unchecked<3>();
    const int n = static_cast<int>(stack.shape(0));
    const int d = static_cast<int>(stack.shape(1));
    Eigen::MatrixXd out(d * d, n);
    for ( int vv = 0; vv < n; ++vv )
    {
        for ( int cc = 0; cc < d; ++cc )
        {
            for ( int rr = 0; rr < d; ++rr )
            {
                out(rr + cc * d, vv) = A(vv, rr, cc);
            }
        }
    }
    return out;
}

// (d*d, n) flat field -> (n, d, d) numpy stack.
py::array_t<double> stack_from_flat_field( const Eigen::MatrixXd& flat, int d )
{
    const int n = static_cast<int>(flat.cols());
    py::array_t<double> out({n, d, d});
    auto A = out.mutable_unchecked<3>();
    for ( int ii = 0; ii < n; ++ii )
    {
        for ( int cc = 0; cc < d; ++cc )
        {
            for ( int rr = 0; rr < d; ++rr )
            {
                A(ii, rr, cc) = flat(rr + cc * d, ii);
            }
        }
    }
    return out;
}

// Ellipsoid list -> (centers (k, d), Sigmas (k, d, d)) numpy pair.
std::pair<Eigen::MatrixXd, py::array_t<double>>
arrays_from_ellipsoids( const std::vector<etree::Ellipsoid>& ells, int d )
{
    const int k = static_cast<int>(ells.size());
    Eigen::MatrixXd centers(k, d);
    py::array_t<double> Sigmas({k, d, d});
    auto A = Sigmas.mutable_unchecked<3>();
    for ( int ii = 0; ii < k; ++ii )
    {
        centers.row(ii) = ells[ii].mu.transpose();
        for ( int rr = 0; rr < d; ++rr )
        {
            for ( int cc = 0; cc < d; ++cc )
            {
                A(ii, rr, cc) = ells[ii].Sigma(rr, cc);
            }
        }
    }
    return std::make_pair(std::move(centers), std::move(Sigmas));
}

// one matrix per entry -> (n, d, d) numpy stack.
py::array_t<double> stack_from_matrices( const std::vector<Eigen::MatrixXd>& mats, int d )
{
    const int n = static_cast<int>(mats.size());
    py::array_t<double> out({n, d, d});
    auto A = out.mutable_unchecked<3>();
    for ( int ii = 0; ii < n; ++ii )
    {
        for ( int rr = 0; rr < d; ++rr )
        {
            for ( int cc = 0; cc < d; ++cc )
            {
                A(ii, rr, cc) = mats[ii](rr, cc);
            }
        }
    }
    return out;
}

const char* frame_name( Frame f )
{
    switch ( f )
    {
        case Frame::identity:         return "identity";
        case Frame::translation:      return "translation";
        case Frame::mean_translation: return "mean_translation";
        case Frame::whitened_affine:  return "whitened_affine";
    }
    return "?";
}

const char* scaling_name( Scaling s )
{
    switch ( s )
    {
        case Scaling::none:       return "none";
        case Scaling::volume:     return "volume";
        case Scaling::volume_det: return "volume_det";
    }
    return "?";
}

const char* support_name( Support s )
{
    switch ( s )
    {
        case Support::none:      return "none";
        case Support::ellipsoid: return "ellipsoid";
    }
    return "?";
}

const char* kernel_name( RBFKernel k )
{
    switch ( k )
    {
        case RBFKernel::gaussian:             return "gaussian";
        case RBFKernel::multiquadric:         return "multiquadric";
        case RBFKernel::inverse_multiquadric: return "inverse_multiquadric";
        case RBFKernel::linear:               return "linear";
        case RBFKernel::thin_plate_spline:    return "thin_plate_spline";
        case RBFKernel::cubic:                return "cubic";
    }
    return "?";
}

} // end anonymous namespace

PYBIND11_MODULE(psfi, m)
{
    m.doc() = "psfi: point spread function interpolation — evaluate an integral kernel\n"
              "Phi(y, x) anywhere by interpolating transported impulse responses sampled\n"
              "at scattered points.\n\n"
              "Array convention: points are rows — point sets are (n, d), mesh vertices\n"
              "(num_vertices, d), mesh cells (num_cells, d+1), covariance stacks (n, d, d).";
    m.attr("__version__") = py::str(std::to_string(PSFI_VERSION_MAJOR) + "."
                                    + std::to_string(PSFI_VERSION_MINOR) + "."
                                    + std::to_string(PSFI_VERSION_PATCH));

    py::enum_<Frame>(m, "Frame",
        "Frame map T_i transporting a stored impulse response at x_i to the query point x.")
        .value("identity", Frame::identity, "T_i(y) = y (paper eq. 4.7)")
        .value("translation", Frame::translation, "T_i(y) = y - x + x_i (4.8)")
        .value("mean_translation", Frame::mean_translation, "T_i(y) = y - mu(x) + mu_i (4.9/4.10)")
        .value("whitened_affine", Frame::whitened_affine,
               "T_i(y) = mu_i + Sigma_i^{1/2} Sigma(x)^{-1/2} (y - mu(x))");

    py::enum_<Scaling>(m, "Scaling",
        "Scalar correction s_i applied to the transported impulse response value.")
        .value("none", Scaling::none, "s_i = 1")
        .value("volume", Scaling::volume, "s_i = V(x)/V_i (preserves peak values)")
        .value("volume_det", Scaling::volume_det,
               "s_i = (V(x)/V_i) sqrt(det Sigma_i / det Sigma(x)) (preserves mass)");

    py::enum_<Support>(m, "Support",
        "Support gate for transported points outside the sample's ellipsoid.")
        .value("none", Support::none, "no gate")
        .value("ellipsoid", Support::ellipsoid, "f_i = 0 outside E_i(tau)");

    py::class_<EvalConfig>(m, "EvalConfig",
        "Evaluation configuration: frame map, scaling, support gate, tau, num_neighbors.\n"
        "Different configurations require different data; see ImpulseResponseField.validate.")
        .def(py::init([]( Frame frame, Scaling scaling, Support support, double tau, int num_neighbors )
             {
                 EvalConfig cfg;
                 cfg.frame = frame;
                 cfg.scaling = scaling;
                 cfg.support = support;
                 cfg.tau = tau;
                 cfg.num_neighbors = num_neighbors;
                 return cfg;
             }),
             "frame"_a = Frame::mean_translation, "scaling"_a = Scaling::volume,
             "support"_a = Support::ellipsoid, "tau"_a = 3.0, "num_neighbors"_a = 10)
        .def_readwrite("frame", &EvalConfig::frame)
        .def_readwrite("scaling", &EvalConfig::scaling)
        .def_readwrite("support", &EvalConfig::support)
        .def_readwrite("tau", &EvalConfig::tau)
        .def_readwrite("num_neighbors", &EvalConfig::num_neighbors)
        .def("__repr__", []( const EvalConfig& c )
             {
                 return std::string("EvalConfig(frame=") + frame_name(c.frame)
                     + ", scaling=" + scaling_name(c.scaling)
                     + ", support=" + support_name(c.support)
                     + ", tau=" + std::to_string(c.tau)
                     + ", num_neighbors=" + std::to_string(c.num_neighbors) + ")";
             });

    py::class_<ImpulseResponseField, std::shared_ptr<ImpulseResponseField>>(m, "ImpulseResponseField",
        "Batches of sampled impulse responses on a simplicial mesh; produces\n"
        "per-neighbor kernel predictions for arbitrary target pairs (y, x).\n\n"
        "vertices: (num_vertices, d); cells: (num_cells, d+1) — the TARGET mesh,\n"
        "carrying the batch functions. Optionally pass source_vertices/source_cells\n"
        "for a separate SOURCE mesh carrying the moment fields and locating the\n"
        "query point x (its dimension may differ; see the moment-map discussion in\n"
        "the C++ docs). By default source = target.\n"
        "batches_normalized=True means each stored batch is sum_i phi_i / V_i\n"
        "(the paper's convention); False means batches store raw impulse responses.")
        .def(py::init([]( const RowsXd& vertices, const RowsXi& cells,
                          bool batches_normalized, int num_threads,
                          std::optional<Eigen::MatrixXd> source_vertices,
                          std::optional<Eigen::MatrixXi> source_cells )
             {
                 if ( source_vertices.has_value() != source_cells.has_value() )
                 {
                     throw std::invalid_argument("provide both source_vertices and source_cells, "
                                                 "or neither");
                 }
                 if ( source_vertices )
                 {
                     return ImpulseResponseField(
                         cols_from_rows(vertices), icols_from_rows(cells),
                         source_vertices->transpose(), source_cells->transpose(),
                         batches_normalized, num_threads);
                 }
                 return ImpulseResponseField(cols_from_rows(vertices), icols_from_rows(cells),
                                             batches_normalized, num_threads);
             }),
             "vertices"_a, "cells"_a, "batches_normalized"_a = true, "num_threads"_a = 0,
             "source_vertices"_a = py::none(), "source_cells"_a = py::none())
        .def_property_readonly("dim_source", &ImpulseResponseField::dim_source)
        .def_property_readonly("dim_target", &ImpulseResponseField::dim_target)
        .def_property_readonly("num_source_vertices", &ImpulseResponseField::num_source_vertices)
        .def_property_readonly("num_target_vertices", &ImpulseResponseField::num_target_vertices)
        .def_property_readonly("has_separate_source_mesh",
                               &ImpulseResponseField::has_separate_source_mesh)
        .def_property_readonly("num_batches", &ImpulseResponseField::num_batches)
        .def_property_readonly("num_sample_points", &ImpulseResponseField::num_sample_points)
        .def_property_readonly("batches_normalized", &ImpulseResponseField::batches_normalized)
        .def_property_readonly("has_sample_V", &ImpulseResponseField::has_sample_V)
        .def_property_readonly("has_sample_mu", &ImpulseResponseField::has_sample_mu)
        .def_property_readonly("has_sample_Sigma", &ImpulseResponseField::has_sample_Sigma)
        .def_property_readonly("has_field_V", &ImpulseResponseField::has_field_V)
        .def_property_readonly("has_field_mu", &ImpulseResponseField::has_field_mu)
        .def_property_readonly("has_field_Sigma", &ImpulseResponseField::has_field_Sigma)
        .def_property_readonly("target_mesh_vertices",
             []( const ImpulseResponseField& F )
             { return Eigen::MatrixXd(F.target_mesh().vertices().transpose()); },
             "Target-mesh vertex coordinates, shape (num_target_vertices, dim_target).")
        .def_property_readonly("target_mesh_cells",
             []( const ImpulseResponseField& F )
             { return Eigen::MatrixXi(F.target_mesh().cells().transpose()); },
             "Target-mesh cell vertex indices, shape (num_cells, dim_target+1).")
        .def_property_readonly("source_mesh_vertices",
             []( const ImpulseResponseField& F )
             { return Eigen::MatrixXd(F.source_mesh().vertices().transpose()); },
             "Source-mesh vertex coordinates (the target mesh's in the square case).")
        .def_property_readonly("source_mesh_cells",
             []( const ImpulseResponseField& F )
             { return Eigen::MatrixXi(F.source_mesh().cells().transpose()); },
             "Source-mesh cell vertex indices (the target mesh's in the square case).")
        .def_property_readonly("sample_points",
             []( const ImpulseResponseField& F )
             { return Eigen::MatrixXd(F.sample_points().transpose()); },
             "Sample points, shape (num_sample_points, d).")
        .def_property_readonly("sample_V",
             []( const ImpulseResponseField& F )
             {
                 return Eigen::Map<const Eigen::VectorXd>(F.sample_V().data(),
                                                          static_cast<int>(F.sample_V().size()))
                     .eval();
             },
             "Per-sample masses V_i, shape (num_sample_points,); empty if absent.")
        .def_property_readonly("sample_mu",
             []( const ImpulseResponseField& F )
             {
                 const auto& mu = F.sample_mu();
                 Eigen::MatrixXd out(static_cast<int>(mu.size()), F.dim_target());
                 for ( int ii = 0; ii < static_cast<int>(mu.size()); ++ii )
                 {
                     out.row(ii) = mu[ii].transpose();
                 }
                 return out;
             },
             "Per-sample means mu_i, shape (num_sample_points, d); empty if absent.")
        .def_property_readonly("sample_Sigma",
             []( const ImpulseResponseField& F )
             { return stack_from_matrices(F.sample_Sigma(), F.dim_target()); },
             "Per-sample covariances Sigma_i (symmetrized), shape (num_sample_points, d, d); "
             "empty if absent.")
        .def_property_readonly("point2batch",
             []( const ImpulseResponseField& F )
             {
                 return Eigen::Map<const Eigen::VectorXi>(F.point2batch().data(),
                                                          static_cast<int>(F.point2batch().size()))
                     .eval();
             },
             "Batch index of each sample point, shape (num_sample_points,).")
        .def("batch_range", &ImpulseResponseField::batch_range, "b"_a,
             "Sample index range (start, stop) of batch b.")
        .def("batch_values", &ImpulseResponseField::batch_values, "b"_a,
             "CG1 vertex values of batch b, shape (num_vertices,).")
        .def("set_moment_fields",
             []( ImpulseResponseField& F,
                 std::optional<Eigen::VectorXd> V,
                 std::optional<Eigen::MatrixXd> mu,
                 std::optional<py::array_t<double>> Sigma )
             {
                 const Eigen::VectorXd V_arg  = V ? *V : Eigen::VectorXd();
                 const Eigen::MatrixXd mu_arg = mu ? cols_from_rows(*mu) : Eigen::MatrixXd();
                 const Eigen::MatrixXd Sigma_arg =
                     Sigma ? flat_field_from_stack(*Sigma, "Sigma") : Eigen::MatrixXd();
                 F.set_moment_fields(V_arg, mu_arg, Sigma_arg);
             },
             "V"_a = py::none(), "mu"_a = py::none(), "Sigma"_a = py::none(),
             "Sets the vertex moment fields; pass None for fields you do not have.\n"
             "V: (num_vertices,); mu: (num_vertices, d); Sigma: (num_vertices, d, d).")
        .def("add_batch",
             []( ImpulseResponseField& F,
                 const RowsXd& points,
                 const Eigen::VectorXd& psi,
                 std::optional<Eigen::VectorXd> V,
                 std::optional<Eigen::MatrixXd> mu,
                 std::optional<py::array_t<double>> Sigma,
                 bool rebuild )
             {
                 const Eigen::VectorXd V_arg  = V ? *V : Eigen::VectorXd();
                 const Eigen::MatrixXd mu_arg = mu ? cols_from_rows(*mu) : Eigen::MatrixXd();
                 const std::vector<Eigen::MatrixXd> Sigma_arg =
                     Sigma ? matrices_from_stack(*Sigma, "Sigma") : std::vector<Eigen::MatrixXd>();
                 F.add_batch(cols_from_rows(points), psi, V_arg, mu_arg, Sigma_arg, rebuild);
             },
             "points"_a, "psi"_a, "V"_a = py::none(), "mu"_a = py::none(), "Sigma"_a = py::none(),
             "rebuild"_a = true,
             "Adds one impulse response batch. points: (num_batch_points, d) arbitrary\n"
             "coordinates; psi: (num_vertices,) CG1 vertex values; optional per-sample\n"
             "moments V (num_batch_points,), mu (num_batch_points, d),\n"
             "Sigma (num_batch_points, d, d). Which moments are supplied is fixed by the\n"
             "first batch. With rebuild=False call rebuild_kdtree() before predictions().")
        .def("rebuild_kdtree", &ImpulseResponseField::rebuild_kdtree)
        .def("validate", &ImpulseResponseField::validate, "config"_a,
             "Raises ValueError listing every piece of data `config` needs but the field\n"
             "does not have; returns None iff predictions(y, x, config) can run.")
        .def("predictions",
             []( const ImpulseResponseField& F,
                 const Eigen::VectorXd& y,
                 const Eigen::VectorXd& x,
                 const EvalConfig& config )
             {
                 const std::vector<Prediction> P = F.predictions(y, x, config);
                 const int k = static_cast<int>(P.size());
                 Eigen::VectorXi indices(k);
                 Eigen::MatrixXd points(k, F.dim_source());
                 Eigen::VectorXd values(k);
                 for ( int jj = 0; jj < k; ++jj )
                 {
                     indices(jj)    = P[jj].sample_index;
                     points.row(jj) = P[jj].point.transpose();
                     values(jj)     = P[jj].value;
                 }
                 return std::make_tuple(indices, points, values);
             },
             "y"_a, "x"_a, "config"_a, py::call_guard<py::gil_scoped_release>(),
             "Per-neighbor kernel predictions at the target pair (y, x), both arbitrary\n"
             "coordinates. Returns (sample_indices (k,), sample_points (k, d),\n"
             "values (k,)), nearest sample first; k <= num_neighbors (samples whose\n"
             "transported point leaves the mesh are excluded, and k = 0 when the\n"
             "configuration needs moment fields at an x outside the mesh).")
        .def("support_ellipsoids",
             []( const ImpulseResponseField& F, const Eigen::VectorXd& x, const EvalConfig& config )
             {
                 return arrays_from_ellipsoids(F.support_ellipsoids(x, config), F.dim_target());
             },
             "x"_a, "config"_a,
             "Ellipsoids covering the support of the predictions at source query x:\n"
             "predictions(y, x, config) are all zero for y outside every ellipsoid\n"
             "(exact, not heuristic). Returns (centers (k, d_target), Sigmas\n"
             "(k, d_target, d_target)) at UNIT scale — config.tau is folded in, so\n"
             "membership is (y - c) @ inv(S) @ (y - c) <= 1. Empty when the column\n"
             "at x is identically zero. Requires Support.ellipsoid.");

    // ------------------------------------------------------------------
    //  Moment-data hygiene
    // ------------------------------------------------------------------

    const char* clamp_spd_doc =
        "Symmetrize a covariance stack (n, d, d) and clamp its eigenvalues to at\n"
        "least `floor` (> 0; scalar or per-entry array of shape (n,)). Returns\n"
        "(cleaned_stack, modified_indices). psfi requires strictly positive\n"
        "definite covariances (add_batch and set_moment_fields validate); use this\n"
        "to repair fields corrupted by numerical error. The floor is a modelling\n"
        "choice, not just hygiene — near-singular covariances pass validation but\n"
        "amplify through Sigma(x)^{-1/2} and det Sigma(x); the square of the local\n"
        "mesh spacing is a reasonable default.";
    auto clamp_stack = []( const py::array_t<double>& Sigmas, const Eigen::VectorXd& floors )
    {
        const Eigen::MatrixXd flat = flat_field_from_stack(Sigmas, "Sigmas");
        const int d = static_cast<int>(Sigmas.shape(1));
        std::pair<Eigen::MatrixXd, std::vector<int>> result = clamp_spd_field(flat, d, floors);
        Eigen::VectorXi modified(static_cast<int>(result.second.size()));
        for ( int ii = 0; ii < modified.size(); ++ii )
        {
            modified(ii) = result.second[ii];
        }
        return std::make_pair(stack_from_flat_field(result.first, d), modified);
    };
    m.def("clamp_spd",
          [clamp_stack]( const py::array_t<double>& Sigmas, double floor )
          {
              return clamp_stack(Sigmas,
                                 Eigen::VectorXd::Constant(Sigmas.shape(0), floor));
          },
          "Sigmas"_a, "floor"_a, clamp_spd_doc);
    m.def("clamp_spd", clamp_stack, "Sigmas"_a, "floor"_a, clamp_spd_doc);

    // ------------------------------------------------------------------
    //  Low-rank tools
    // ------------------------------------------------------------------
    // Generic matrix layer: plain rows/columns, no source/target semantics
    // (see the low_rank.hpp file header). Matrices pass through unchanged —
    // the points-are-rows transposition does not apply here.

    py::class_<LowRank>(m, "LowRank",
        "Rank-r factorization A ~ U @ V.T with U (num_rows, r), V (num_cols, r).\n"
        "Scale is folded into U; V is orthonormal when produced by\n"
        "truncated_svd / recompress / randomized_svd.")
        .def(py::init([]( const Eigen::MatrixXd& U, const Eigen::MatrixXd& V )
             {
                 if ( U.cols() != V.cols() )
                 {
                     throw std::invalid_argument("LowRank: U and V must have the same number "
                                                 "of columns (the rank)");
                 }
                 return LowRank{ U, V };
             }),
             "U"_a, "V"_a)
        .def_readwrite("U", &LowRank::U)
        .def_readwrite("V", &LowRank::V)
        .def_property_readonly("rank", &LowRank::rank)
        .def("to_dense", &LowRank::to_dense, "The dense matrix U @ V.T.")
        .def("__repr__", []( const LowRank& F )
             {
                 return "LowRank(num_rows=" + std::to_string(F.U.rows())
                     + ", num_cols=" + std::to_string(F.V.rows())
                     + ", rank=" + std::to_string(F.rank()) + ")";
             });

    m.def("truncated_svd",
          []( const Eigen::MatrixXd& A, double rtol, int max_rank )
          { return truncated_svd(A, rtol, max_rank); },
          "A"_a, "rtol"_a, "max_rank"_a = -1,
          py::call_guard<py::gil_scoped_release>(),
          "Best approximation with the smallest rank whose discarded singular-value\n"
          "tail satisfies ||tail||_F <= rtol * ||A||_F, capped at max_rank\n"
          "(max_rank < 0 means no cap).");

    m.def("recompress",
          []( const LowRank& F, double rtol, int max_rank )
          { return recompress(F, rtol, max_rank); },
          "factors"_a, "rtol"_a, "max_rank"_a = -1,
          py::call_guard<py::gil_scoped_release>(),
          "Recompresses a (possibly redundant) factorization to the smallest rank\n"
          "meeting the relative Frobenius tolerance.");

    py::class_<ACAOptions>(m, "ACAOptions",
        "Options for aca(). The stopping tolerance is aca_safety_factor * rtol and\n"
        "the recompression tolerance recompress_safety_factor * rtol.")
        .def(py::init([]( int max_rank, double aca_safety_factor, double recompress_safety_factor,
                          int required_consecutive_successes, bool recompress, unsigned int seed )
             {
                 ACAOptions o;
                 o.max_rank = max_rank;
                 o.aca_safety_factor = aca_safety_factor;
                 o.recompress_safety_factor = recompress_safety_factor;
                 o.required_consecutive_successes = required_consecutive_successes;
                 o.recompress = recompress;
                 o.seed = seed;
                 return o;
             }),
             "max_rank"_a = -1, "aca_safety_factor"_a = 0.25, "recompress_safety_factor"_a = 0.75,
             "required_consecutive_successes"_a = 10, "recompress"_a = true, "seed"_a = 0)
        .def_readwrite("max_rank", &ACAOptions::max_rank)
        .def_readwrite("aca_safety_factor", &ACAOptions::aca_safety_factor)
        .def_readwrite("recompress_safety_factor", &ACAOptions::recompress_safety_factor)
        .def_readwrite("required_consecutive_successes", &ACAOptions::required_consecutive_successes)
        .def_readwrite("recompress", &ACAOptions::recompress)
        .def_readwrite("seed", &ACAOptions::seed)
        .def("__repr__", []( const ACAOptions& o )
             {
                 return "ACAOptions(max_rank=" + std::to_string(o.max_rank)
                     + ", aca_safety_factor=" + std::to_string(o.aca_safety_factor)
                     + ", recompress_safety_factor=" + std::to_string(o.recompress_safety_factor)
                     + ", required_consecutive_successes="
                     + std::to_string(o.required_consecutive_successes)
                     + ", recompress=" + ( o.recompress ? "True" : "False" )
                     + ", seed=" + std::to_string(o.seed) + ")";
             });

    py::class_<ACAResult>(m, "ACAResult",
        "aca() result: factors plus construction diagnostics. Check hit_max_rank —\n"
        "never let a rank cap bind silently.")
        .def_readonly("factors", &ACAResult::factors)
        .def_readonly("sampled_rows", &ACAResult::sampled_rows)
        .def_readonly("sampled_cols", &ACAResult::sampled_cols)
        .def_readonly("sampled_rank", &ACAResult::sampled_rank)
        .def_readonly("converged", &ACAResult::converged)
        .def_readonly("hit_max_rank", &ACAResult::hit_max_rank)
        .def_readonly("relerr_estimate", &ACAResult::relerr_estimate)
        .def("__repr__", []( const ACAResult& r )
             {
                 return "ACAResult(rank=" + std::to_string(r.factors.rank())
                     + ", sampled_rank=" + std::to_string(r.sampled_rank)
                     + ", converged=" + ( r.converged ? "True" : "False" )
                     + ", hit_max_rank=" + ( r.hit_max_rank ? "True" : "False" ) + ")";
             });

    m.def("aca",
          []( const std::function<Eigen::VectorXd(int)>& get_row,
              const std::function<Eigen::VectorXd(int)>& get_col,
              int num_rows, int num_cols, double rtol, const ACAOptions& options )
          { return aca(get_row, get_col, num_rows, num_cols, rtol, options); },
          "get_row"_a, "get_col"_a, "num_rows"_a, "num_cols"_a, "rtol"_a,
          "options"_a = ACAOptions{},
          "Adaptive cross approximation with partial pivoting and random-restart\n"
          "verification (the GPSF/ymir 'ACA+'). get_row(i) must return row i and\n"
          "get_col(j) column j of the same fixed matrix; only sampled rows and\n"
          "columns are ever evaluated. rtol = 0 runs to exact recovery.\n"
          "Deterministic for a given options.seed.");

    py::class_<RSVDOptions>(m, "RSVDOptions", "Options for randomized_svd().")
        .def(py::init([]( int oversampling, int power_iterations, double rtol, unsigned int seed )
             {
                 RSVDOptions o;
                 o.oversampling = oversampling;
                 o.power_iterations = power_iterations;
                 o.rtol = rtol;
                 o.seed = seed;
                 return o;
             }),
             "oversampling"_a = 10, "power_iterations"_a = 1, "rtol"_a = 0.0, "seed"_a = 0)
        .def_readwrite("oversampling", &RSVDOptions::oversampling)
        .def_readwrite("power_iterations", &RSVDOptions::power_iterations)
        .def_readwrite("rtol", &RSVDOptions::rtol)
        .def_readwrite("seed", &RSVDOptions::seed)
        .def("__repr__", []( const RSVDOptions& o )
             {
                 return "RSVDOptions(oversampling=" + std::to_string(o.oversampling)
                     + ", power_iterations=" + std::to_string(o.power_iterations)
                     + ", rtol=" + std::to_string(o.rtol)
                     + ", seed=" + std::to_string(o.seed) + ")";
             });

    m.def("randomized_svd",
          []( const std::function<Eigen::MatrixXd(const Eigen::Ref<const Eigen::MatrixXd>&)>& apply,
              const std::function<Eigen::MatrixXd(const Eigen::Ref<const Eigen::MatrixXd>&)>& apply_transpose,
              int num_rows, int num_cols, int max_rank, const RSVDOptions& options )
          { return randomized_svd(apply, apply_transpose, num_rows, num_cols, max_rank, options); },
          "apply"_a, "apply_transpose"_a, "num_rows"_a, "num_cols"_a, "max_rank"_a,
          "options"_a = RSVDOptions{},
          "Randomized SVD of a matrix available only through its action:\n"
          "apply(X) = A @ X on (num_cols, k) blocks, apply_transpose(Y) = A.T @ Y on\n"
          "(num_rows, k) blocks. Roughly the best rank-max_rank approximation;\n"
          "deterministic for a given options.seed.");

    // ------------------------------------------------------------------
    //  RBF interpolation
    // ------------------------------------------------------------------

    py::enum_<RBFKernel>(m, "RBFKernel",
        "Radial kernel phi(u) at the locally scaled distance u = shape * r / r0\n"
        "(r0 = diameter of the interpolation point set, recomputed per call).\n"
        "Sign conventions follow scipy.interpolate.RBFInterpolator.")
        .value("gaussian", RBFKernel::gaussian, "exp(-u^2/2)")
        .value("multiquadric", RBFKernel::multiquadric, "-sqrt(1 + u^2)")
        .value("inverse_multiquadric", RBFKernel::inverse_multiquadric, "1/sqrt(1 + u^2)")
        .value("linear", RBFKernel::linear, "-u")
        .value("thin_plate_spline", RBFKernel::thin_plate_spline, "u^2 log u")
        .value("cubic", RBFKernel::cubic, "u^3");

    py::class_<RBFScheme>(m, "RBFScheme",
        "RBF interpolation configuration: kernel, shape (the paper's C_RBF),\n"
        "polynomial-tail degree (-1 none, 0 constant, 1 linear), ridge smoothing.\n"
        "With smoothing = 0 the interpolant reproduces its data at the centers;\n"
        "smoothing has no effect when there are no more centers than tail\n"
        "coefficients (the system degenerates to polynomial interpolation).")
        .def(py::init([]( RBFKernel kernel, double shape, int degree, double smoothing )
             {
                 RBFScheme s;
                 s.kernel = kernel;
                 s.shape = shape;
                 s.degree = degree;
                 s.smoothing = smoothing;
                 validate(s);
                 return s;
             }),
             "kernel"_a = RBFKernel::gaussian, "shape"_a = 3.0, "degree"_a = 1, "smoothing"_a = 0.0)
        .def_readwrite("kernel", &RBFScheme::kernel)
        .def_readwrite("shape", &RBFScheme::shape)
        .def_readwrite("degree", &RBFScheme::degree)
        .def_readwrite("smoothing", &RBFScheme::smoothing)
        .def("__repr__", []( const RBFScheme& s )
             {
                 return std::string("RBFScheme(kernel=") + kernel_name(s.kernel)
                     + ", shape=" + std::to_string(s.shape)
                     + ", degree=" + std::to_string(s.degree)
                     + ", smoothing=" + std::to_string(s.smoothing) + ")";
             });

    m.def("rbf_min_degree", &rbf_min_degree, "kernel"_a,
          "Smallest polynomial-tail degree guaranteeing solvability for this kernel.");

    m.def("rbf_interpolate",
          []( const Eigen::VectorXd& values, const RowsXd& centers, const RowsXd& eval_points,
              const RBFScheme& scheme )
          {
              return rbf_interpolate(values, cols_from_rows(centers).eval(),
                                     cols_from_rows(eval_points).eval(), scheme);
          },
          "values"_a, "centers"_a, "eval_points"_a, "scheme"_a = RBFScheme{},
          py::call_guard<py::gil_scoped_release>(),
          "RBF interpolant of {(centers[i], values[i])} evaluated at eval_points.\n"
          "values: (k,); centers: (k, d); eval_points: (m, d). Returns (m,).\n"
          "One center (or all coincident) gives a constant; the tail degree is\n"
          "lowered automatically when there are fewer centers than coefficients.");

    // ------------------------------------------------------------------
    //  Kernel evaluator
    // ------------------------------------------------------------------

    py::class_<KernelEvaluator>(m, "KernelEvaluator",
        "The complete kernel approximation Phi(y, x): RBF combination of\n"
        "per-neighbor predictions. Cols-only with one field, symmetric with a\n"
        "row field probed with the transpose operator (pass the same field\n"
        "twice for a symmetric operator probed once); in symmetric mode the\n"
        "forward and adjoint prediction sets are pooled in displacement\n"
        "coordinates and near-duplicate centers are averaged. Entries at\n"
        "sample columns are exact when smoothing = 0 (center reproduction —\n"
        "no snapping special case).")
        .def(py::init([]( std::shared_ptr<const ImpulseResponseField> col_field,
                          std::shared_ptr<const ImpulseResponseField> row_field,
                          const EvalConfig& config, const RBFScheme& rbf, double duplicate_tol )
             {
                 return KernelEvaluator(std::move(col_field), std::move(row_field),
                                        config, rbf, duplicate_tol);
             }),
             "col_field"_a, "row_field"_a = nullptr, "config"_a = EvalConfig{},
             "rbf"_a = RBFScheme{}, "duplicate_tol"_a = 1e-7,
             "Validates the fields against the configuration here: construction\n"
             "succeeds iff evaluation can run.")
        .def_property_readonly("dim_source", &KernelEvaluator::dim_source)
        .def_property_readonly("dim_target", &KernelEvaluator::dim_target)
        .def_property_readonly("symmetric", &KernelEvaluator::symmetric)
        .def_property_readonly("duplicate_tol", &KernelEvaluator::duplicate_tol)
        .def_property_readonly("config", []( const KernelEvaluator& K ) { return K.config(); })
        .def_property_readonly("rbf", []( const KernelEvaluator& K ) { return K.rbf(); })
        .def("__call__",
             []( const KernelEvaluator& K, const Eigen::VectorXd& y, const Eigen::VectorXd& x )
             { return K(y, x); },
             "y"_a, "x"_a, py::call_guard<py::gil_scoped_release>(),
             "The approximate kernel entry Phi(y, x); zero where there are no predictions.")
        .def("block",
             []( const KernelEvaluator& K, const RowsXd& yy, const RowsXd& xx, int num_threads )
             {
                 return Eigen::MatrixXd(K.block(cols_from_rows(yy).eval(),
                                                cols_from_rows(xx).eval(), num_threads));
             },
             "yy"_a, "xx"_a, "num_threads"_a = 0, py::call_guard<py::gil_scoped_release>(),
             "Block [Phi(yy[i], xx[j])]_{ij} of shape (num_y, num_x), evaluated in\n"
             "parallel; yy: (num_y, d), xx: (num_x, d). num_threads <= 0 uses all cores.")
        .def("target_support",
             []( const KernelEvaluator& K, const Eigen::VectorXd& x )
             { return arrays_from_ellipsoids(K.target_support(x), K.dim_target()); },
             "x"_a,
             "Ellipsoids (unit scale, tau folded in) covering the col-field support\n"
             "of the kernel column Phi(., x), as (centers (k, d_target), Sigmas\n"
             "(k, d_target, d_target)). Covers the whole column in cols-only mode;\n"
             "in symmetric mode the full column support additionally includes the\n"
             "targets y with x inside source_support(y).")
        .def("source_support",
             []( const KernelEvaluator& K, const Eigen::VectorXd& y )
             { return arrays_from_ellipsoids(K.source_support(y), K.dim_source()); },
             "y"_a,
             "Symmetric mode only: ellipsoids in the SOURCE domain covering the\n"
             "row-field contribution to the kernel row Phi(y, .), as (centers\n"
             "(k, d_source), Sigmas (k, d_source, d_source)). Raises in cols-only\n"
             "mode.");

    // ------------------------------------------------------------------
    //  Kernel low rank (the source/target <-> rows/cols adapter)
    // ------------------------------------------------------------------

    py::enum_<CompressionMethod>(m, "CompressionMethod",
        "How to compress a kernel matrix: dense_svd assembles the dense matrix and\n"
        "Frobenius-truncates its SVD (exact, superlinear cost); aca samples\n"
        "O(rank * (num_targets + num_sources)) entries (the production path at\n"
        "scale); automatic picks dense_svd when min(num_targets, num_sources) <=\n"
        "dense_min_dim, else aca.")
        .value("automatic", CompressionMethod::automatic)
        .value("dense_svd", CompressionMethod::dense_svd)
        .value("aca", CompressionMethod::aca);

    py::class_<KernelLowRankOptions>(m, "KernelLowRankOptions",
        "Options for kernel_low_rank(). The ACA knobs mirror ACAOptions\n"
        "(recompression always on).")
        .def(py::init([]( CompressionMethod method, int max_rank, int dense_min_dim,
                          int num_threads, double aca_safety_factor,
                          double recompress_safety_factor, int required_consecutive_successes,
                          unsigned int seed )
             {
                 KernelLowRankOptions o;
                 o.method = method;
                 o.max_rank = max_rank;
                 o.dense_min_dim = dense_min_dim;
                 o.num_threads = num_threads;
                 o.aca_safety_factor = aca_safety_factor;
                 o.recompress_safety_factor = recompress_safety_factor;
                 o.required_consecutive_successes = required_consecutive_successes;
                 o.seed = seed;
                 return o;
             }),
             "method"_a = CompressionMethod::automatic, "max_rank"_a = -1,
             "dense_min_dim"_a = 128, "num_threads"_a = 0, "aca_safety_factor"_a = 0.25,
             "recompress_safety_factor"_a = 0.75, "required_consecutive_successes"_a = 10,
             "seed"_a = 0)
        .def_readwrite("method", &KernelLowRankOptions::method)
        .def_readwrite("max_rank", &KernelLowRankOptions::max_rank)
        .def_readwrite("dense_min_dim", &KernelLowRankOptions::dense_min_dim)
        .def_readwrite("num_threads", &KernelLowRankOptions::num_threads)
        .def_readwrite("aca_safety_factor", &KernelLowRankOptions::aca_safety_factor)
        .def_readwrite("recompress_safety_factor", &KernelLowRankOptions::recompress_safety_factor)
        .def_readwrite("required_consecutive_successes",
                       &KernelLowRankOptions::required_consecutive_successes)
        .def_readwrite("seed", &KernelLowRankOptions::seed)
        .def("__repr__", []( const KernelLowRankOptions& o )
             {
                 const char* method = ( o.method == CompressionMethod::automatic ) ? "automatic"
                     : ( o.method == CompressionMethod::dense_svd ) ? "dense_svd" : "aca";
                 return std::string("KernelLowRankOptions(method=") + method
                     + ", max_rank=" + std::to_string(o.max_rank)
                     + ", dense_min_dim=" + std::to_string(o.dense_min_dim)
                     + ", num_threads=" + std::to_string(o.num_threads)
                     + ", aca_safety_factor=" + std::to_string(o.aca_safety_factor)
                     + ", recompress_safety_factor=" + std::to_string(o.recompress_safety_factor)
                     + ", required_consecutive_successes="
                     + std::to_string(o.required_consecutive_successes)
                     + ", seed=" + std::to_string(o.seed) + ")";
             });

    py::class_<KernelLowRankResult>(m, "KernelLowRankResult",
        "kernel_low_rank() result: factors (U rows = target points, V rows = source\n"
        "points) plus diagnostics. Check hit_max_rank — a binding rank cap is never\n"
        "silent.")
        .def_readonly("factors", &KernelLowRankResult::factors)
        .def_readonly("used_aca", &KernelLowRankResult::used_aca)
        .def_readonly("converged", &KernelLowRankResult::converged)
        .def_readonly("hit_max_rank", &KernelLowRankResult::hit_max_rank)
        .def_readonly("relerr_estimate", &KernelLowRankResult::relerr_estimate)
        .def("__repr__", []( const KernelLowRankResult& r )
             {
                 return "KernelLowRankResult(rank=" + std::to_string(r.factors.rank())
                     + ", used_aca=" + ( r.used_aca ? "True" : "False" )
                     + ", converged=" + ( r.converged ? "True" : "False" )
                     + ", hit_max_rank=" + ( r.hit_max_rank ? "True" : "False" ) + ")";
             });

    m.def("kernel_low_rank",
          []( const KernelEvaluator& kernel, const RowsXd& yy, const RowsXd& xx, double rtol,
              const KernelLowRankOptions& options )
          {
              return kernel_low_rank(kernel, cols_from_rows(yy).eval(),
                                     cols_from_rows(xx).eval(), rtol, options);
          },
          "kernel"_a, "yy"_a, "xx"_a, "rtol"_a, "options"_a = KernelLowRankOptions{},
          py::call_guard<py::gil_scoped_release>(),
          "Low-rank approximation of the kernel matrix [Phi(yy[i], xx[j])]_{ij} to\n"
          "relative Frobenius tolerance rtol; yy: (num_targets, dim_target),\n"
          "xx: (num_sources, dim_source). In the returned factors, U rows\n"
          "correspond to targets and V rows to sources (target axis = matrix rows,\n"
          "source axis = matrix columns, matching KernelEvaluator.block).\n"
          "Deterministic for a given options.seed.");

    // ------------------------------------------------------------------
    //  Partitioning and per-block target sets
    // ------------------------------------------------------------------

    m.def("recursive_bisection_partition",
          []( const RowsXd& points, int max_part_size )
          { return recursive_bisection_partition(cols_from_rows(points).eval(), max_part_size); },
          "points"_a, "max_part_size"_a, py::call_guard<py::gil_scoped_release>(),
          "Partitions {0, ..., n-1} into spatially coherent parts of at most\n"
          "max_part_size points by recursive median bisection along cycling axes;\n"
          "points: (n, d). Returns a list of sorted index lists (disjoint, covering).\n"
          "Deterministic across platforms. A partition is plain data — any other\n"
          "partitioner can produce the same shape for the downstream builders.");

    m.def("block_target_sets",
          []( const KernelEvaluator& kernel, const RowsXd& yy, const RowsXd& xx,
              const std::vector<std::vector<int>>& source_partition, int num_threads )
          {
              return block_target_sets(kernel, cols_from_rows(yy).eval(),
                                       cols_from_rows(xx).eval(), source_partition, num_threads);
          },
          "kernel"_a, "yy"_a, "xx"_a, "source_partition"_a, "num_threads"_a = 0,
          py::call_guard<py::gil_scoped_release>(),
          "For each part of a source partition (disjoint index lists into the rows\n"
          "of xx), the sorted indices of the targets (rows of yy) where kernel\n"
          "entries against that part's sources can be nonzero — every excluded\n"
          "entry is exactly zero, by the support oracles. Symmetric mode includes\n"
          "the adjoint piece; Support.none gives every part all targets.");

    // ------------------------------------------------------------------
    //  Block low rank (the BRLR format)
    // ------------------------------------------------------------------

    py::class_<BlockLowRank> block_low_rank_class(m, "BlockLowRank",
        "The source-partitioned compressed kernel matrix. The operator/transpose\n"
        "dictionary: the impulse field samples columns of the kernel of whatever\n"
        "operator OP was probed, so apply ~ OP (nodal kernel sense; mass matrices\n"
        "stay downstream, A = M Phi M) and applyT ~ OP^T. Probed A forward =>\n"
        "apply ~ A; probed A^T => apply ~ A^T and A itself is applyT; symmetric\n"
        "mode => approximately symmetric, form (apply + applyT)/2 if wanted.\n"
        "Pure linear algebra: global ids + factors + axis sizes, no geometry.");

    py::class_<BlockLowRank::Block>(block_low_rank_class, "Block",
        "One block: the kernel restricted to (target_ids, source_ids). factored:\n"
        "block ~ target_factor (|T|, r) @ source_factor.T; dense: target_factor is\n"
        "the (|T|, |S|) block verbatim and source_factor is empty.")
        .def(py::init([]( std::vector<int> source_ids, std::vector<int> target_ids,
                          bool factored, const Eigen::MatrixXd& target_factor,
                          const Eigen::MatrixXd& source_factor )
             {
                 BlockLowRank::Block blk;
                 blk.source_ids = std::move(source_ids);
                 blk.target_ids = std::move(target_ids);
                 blk.factored = factored;
                 blk.target_factor = target_factor;
                 blk.source_factor = source_factor;
                 return blk;
             }),
             "source_ids"_a, "target_ids"_a, "factored"_a, "target_factor"_a,
             "source_factor"_a = Eigen::MatrixXd(0, 0))
        .def_readonly("source_ids", &BlockLowRank::Block::source_ids)
        .def_readonly("target_ids", &BlockLowRank::Block::target_ids)
        .def_readonly("factored", &BlockLowRank::Block::factored)
        .def_readonly("target_factor", &BlockLowRank::Block::target_factor)
        .def_readonly("source_factor", &BlockLowRank::Block::source_factor)
        .def_property_readonly("rank", &BlockLowRank::Block::rank)
        .def_property_readonly("storage_entries", &BlockLowRank::Block::storage_entries)
        .def("__repr__", []( const BlockLowRank::Block& blk )
             {
                 return "Block(num_sources=" + std::to_string(blk.source_ids.size())
                     + ", num_targets=" + std::to_string(blk.target_ids.size())
                     + ", " + ( blk.factored ? "factored, rank=" + std::to_string(blk.rank())
                                             : std::string("dense") ) + ")";
             });

    block_low_rank_class
        .def(py::init([]( int num_sources, int num_targets,
                          std::vector<BlockLowRank::Block> blocks )
             { return BlockLowRank(num_sources, num_targets, std::move(blocks)); }),
             "num_sources"_a, "num_targets"_a, "blocks"_a,
             "Validates ids and factor shapes; blocks must partition the source axis.")
        .def_property_readonly("num_sources", &BlockLowRank::num_sources)
        .def_property_readonly("num_targets", &BlockLowRank::num_targets)
        .def_property_readonly("num_blocks", &BlockLowRank::num_blocks)
        .def_property_readonly("blocks", &BlockLowRank::blocks)
        .def_property_readonly("storage_entries", &BlockLowRank::storage_entries)
        .def("apply",
             []( const BlockLowRank& B, const Eigen::Ref<const Eigen::MatrixXd>& U )
             { return B.apply(U); },
             "U"_a, py::call_guard<py::gil_scoped_release>(),
             "Integrate against the source axis: U (num_sources, k) -> (num_targets, k).\n"
             "Approximates the probed operator OP (see the class dictionary). For a\n"
             "single vector pass U[:, None].")
        .def("applyT",
             []( const BlockLowRank& B, const Eigen::Ref<const Eigen::MatrixXd>& V )
             { return B.applyT(V); },
             "V"_a, py::call_guard<py::gil_scoped_release>(),
             "The transpose action: V (num_targets, k) -> (num_sources, k) ~ OP^T.")
        .def("to_dense", &BlockLowRank::to_dense,
             py::call_guard<py::gil_scoped_release>(),
             "The dense (num_targets, num_sources) matrix — for tests and small problems.")
        .def("__repr__", []( const BlockLowRank& B )
             {
                 return "BlockLowRank(num_targets=" + std::to_string(B.num_targets())
                     + ", num_sources=" + std::to_string(B.num_sources())
                     + ", num_blocks=" + std::to_string(B.num_blocks())
                     + ", storage_entries=" + std::to_string(B.storage_entries()) + ")";
             });

    py::class_<BlockBuildInfo>(m, "BlockBuildInfo",
        "Per-block construction diagnostics (indices parallel the partition).")
        .def_readonly("used_aca", &BlockBuildInfo::used_aca)
        .def_readonly("converged", &BlockBuildInfo::converged)
        .def_readonly("hit_max_rank", &BlockBuildInfo::hit_max_rank)
        .def_readonly("relerr_estimate", &BlockBuildInfo::relerr_estimate);

    py::class_<BlockLowRankBuildResult>(m, "BlockLowRankBuildResult",
        "block_low_rank() result: the matrix plus diagnostics. Check\n"
        "any_hit_max_rank — a binding rank cap is never silent.")
        .def_readonly("matrix", &BlockLowRankBuildResult::matrix)
        .def_readonly("block_info", &BlockLowRankBuildResult::block_info)
        .def_readonly("all_converged", &BlockLowRankBuildResult::all_converged)
        .def_readonly("any_hit_max_rank", &BlockLowRankBuildResult::any_hit_max_rank)
        .def("__repr__", []( const BlockLowRankBuildResult& r )
             {
                 return "BlockLowRankBuildResult(num_blocks="
                     + std::to_string(r.matrix.num_blocks())
                     + ", all_converged=" + ( r.all_converged ? "True" : "False" )
                     + ", any_hit_max_rank=" + ( r.any_hit_max_rank ? "True" : "False" ) + ")";
             });

    m.def("block_low_rank",
          []( const KernelEvaluator& kernel, const RowsXd& yy, const RowsXd& xx,
              const std::vector<std::vector<int>>& source_partition, double rtol,
              const KernelLowRankOptions& options )
          {
              return block_low_rank(kernel, cols_from_rows(yy).eval(),
                                    cols_from_rows(xx).eval(), source_partition, rtol, options);
          },
          "kernel"_a, "yy"_a, "xx"_a, "source_partition"_a, "rtol"_a,
          "options"_a = KernelLowRankOptions{},
          py::call_guard<py::gil_scoped_release>(),
          "Builds the BRLR approximation of the kernel matrix: target sets from the\n"
          "support oracles (lossless block sparsity), per-block compression\n"
          "(dense-SVD or ACA per options), then the smaller of factored and\n"
          "verbatim-dense storage per block (dense-path dense storage keeps the\n"
          "EXACT block). rtol is the per-block relative Frobenius tolerance, which\n"
          "is also the whole-matrix bound. yy: (num_targets, dim_target),\n"
          "xx: (num_sources, dim_source). Deterministic (per-block seeds\n"
          "options.seed + block index).");
}

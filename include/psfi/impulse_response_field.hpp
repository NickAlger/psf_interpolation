#pragma once
// SPDX-License-Identifier: MIT
// Part of psfi — https://github.com/NickAlger/psf_interpolation

/// @file
/// @brief ImpulseResponseField: batches of sampled impulse responses on a
/// simplicial mesh, with the per-neighbor kernel predictions that the RBF
/// layer interpolates.
///
/// Data model. For an operator A : L^2(Omega_source) -> L^2(Omega_target)'
/// with kernel Phi(y, x), the impulse response phi_x = Phi(., x) is a
/// function ON the target domain, INDEXED by points x of the source domain
/// (the two domains coincide for the classical square case, and may have
/// different dimensions otherwise). Every piece of data has one of two
/// homes:
///
///  - target side (function data): a simplicial mesh (etree::SimplexMesh)
///    carrying the batch functions as CG1 vertex vectors — each batch is the
///    response of the operator to a weighted Dirac comb — plus all moment
///    VALUES: masses V in R, means mu in Omega_target, covariances Sigma in
///    SPD(dim_target);
///  - source side (indexing): the sample points x_i (arbitrary coordinates,
///    not necessarily mesh vertices), the query point x, the kd-tree over
///    the samples, and an optional second mesh used for exactly one thing —
///    CG1-interpolating the moment map at x.
///
/// The moment map M : Omega_source -> R x Omega_target x SPD(dim_target),
/// x -> (V(x), mu(x), Sigma(x)), is stored as vertex fields over the SOURCE
/// mesh with TARGET-valued entries; mu is literally a map from the source
/// domain into the target domain, which is what makes kernels between
/// different domains (and dimensions) work. By default the source mesh is
/// the target mesh and nothing distinguishes the two roles.
///
/// All per-sample moments and both batch-normalization conventions are
/// optional; ImpulseResponseField::validate reports exactly what a given
/// EvalConfig needs. See config.hpp for the frame map / scaling / support
/// axes.
///
/// Locality. Each mesh is *a* mesh, not necessarily "the whole domain": in a
/// distributed setting each rank holds submeshes (plus halo) and the samples
/// whose support ellipsoids reach them. All queries are answered from local
/// data only; "outside the mesh" is treated as "outside the domain" (the
/// kernel is zero there). Callers doing domain decomposition must provide
/// enough halo that this identification is correct for the queries they
/// issue. The domain-membership test is isolated in locate() so that a
/// separate global-domain indicator can plug in without touching the
/// evaluation logic.
///
/// Values are scalar today; a vector/tensor-valued extension is planned and
/// verified additive — see dev/VECTOR_TENSOR_EXTENSION.md, including the
/// geometry/value-separation invariant that keeps it cheap.

#include <cmath>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Dense>

#include "etree/kd_tree.hpp"
#include "etree/simplex_mesh.hpp"

#include "psfi/config.hpp"

namespace psfi {

/// One neighbor's prediction of the kernel value at (y, x): sample x_i
/// predicts f_i = s_i * phi_i(T_i(y)). The RBF layer interpolates the pairs
/// {(x_i, f_i)} at x (in source-domain coordinates).
struct Prediction
{
    int             sample_index; ///< index into the field's sample points
    Eigen::VectorXd point;        ///< the sample point x_i (source domain)
    double          value;        ///< the predicted kernel value f_i
};

/// Batches of sampled impulse responses on a simplicial mesh; produces
/// per-neighbor kernel predictions for arbitrary target pairs (y, x).
class ImpulseResponseField
{
public:
    /// Square case: source domain = target domain. Builds the field on a
    /// d-simplex mesh from vertex coordinates (dim, num_vertices) and cell
    /// vertex-index columns (dim+1, num_cells). batches_normalized declares
    /// the stored batch convention: true means each batch is sum_i phi_i/V_i
    /// (the paper's Dirac combs are weighted by 1/V_i), false means batches
    /// store raw impulse responses sum_i phi_i.
    ImpulseResponseField( const Eigen::Ref<const Eigen::MatrixXd>& target_vertices,
                          const Eigen::Ref<const Eigen::MatrixXi>& target_cells,
                          bool batches_normalized = true,
                          int  num_threads = 0 )
        : mesh_target_(target_vertices, target_cells, num_threads),
          batches_normalized_(batches_normalized)
    {
        dim_target_ = mesh_target_.dim();
        dim_source_ = dim_target_;
    }

    /// Rectangular case: a separate source mesh carries the moment fields
    /// and locates query points x; its dimension may differ from the
    /// target's (see the moment-map discussion in the file header).
    ImpulseResponseField( const Eigen::Ref<const Eigen::MatrixXd>& target_vertices,
                          const Eigen::Ref<const Eigen::MatrixXi>& target_cells,
                          const Eigen::Ref<const Eigen::MatrixXd>& source_vertices,
                          const Eigen::Ref<const Eigen::MatrixXi>& source_cells,
                          bool batches_normalized = true,
                          int  num_threads = 0 )
        : mesh_target_(target_vertices, target_cells, num_threads),
          mesh_source_(std::in_place, source_vertices, source_cells, num_threads),
          batches_normalized_(batches_normalized)
    {
        dim_target_ = mesh_target_.dim();
        dim_source_ = mesh_source_->dim();
    }

    /// Source-domain dimension (where the sample points and query x live).
    int dim_source() const        { return dim_source_; }
    /// Target-domain dimension (where the impulse responses and query y live).
    int dim_target() const        { return dim_target_; }
    /// Number of target-mesh vertices (the length of a batch vector).
    int num_target_vertices() const { return mesh_target_.num_vertices(); }
    /// Number of source-mesh vertices (the length of the moment fields).
    int num_source_vertices() const { return source_mesh().num_vertices(); }
    /// Number of impulse response batches added so far.
    int num_batches() const       { return static_cast<int>(batch_values_.size()); }
    /// Total number of sample points across all batches.
    int num_sample_points() const { return static_cast<int>(sample_points_.size()); }
    /// The mesh carrying the batch functions.
    const etree::SimplexMesh& target_mesh() const { return mesh_target_; }
    /// The mesh carrying the moment fields (the target mesh in the square case).
    const etree::SimplexMesh& source_mesh() const
    {
        return mesh_source_ ? *mesh_source_ : mesh_target_;
    }
    /// True if a separate source mesh was supplied at construction.
    bool has_separate_source_mesh() const { return mesh_source_.has_value(); }
    /// Batch normalization convention declared at construction.
    bool batches_normalized() const { return batches_normalized_; }

    /// Whether per-sample masses V_i / means mu_i / covariances Sigma_i are
    /// stored (fixed by what the first add_batch supplies).
    bool has_sample_V() const     { return has_sample_V_; }
    bool has_sample_mu() const    { return has_sample_mu_; }
    bool has_sample_Sigma() const { return has_sample_Sigma_; }
    /// Whether the vertex moment fields V / mu / Sigma are set.
    bool has_field_V() const      { return field_V_.size() > 0; }
    bool has_field_mu() const     { return field_mu_.cols() > 0; }
    bool has_field_Sigma() const  { return field_Sigma_.cols() > 0; }

    /// Sample points as columns, shape (dim_source, num_sample_points).
    Eigen::MatrixXd sample_points() const
    {
        Eigen::MatrixXd P(dim_source_, num_sample_points());
        for ( int ii = 0; ii < num_sample_points(); ++ii )
        {
            P.col(ii) = sample_points_[ii];
        }
        return P;
    }
    /// Per-sample masses V_i (empty if absent).
    const std::vector<double>& sample_V() const                 { return sample_V_; }
    /// Per-sample means mu_i in the target domain (empty if absent).
    const std::vector<Eigen::VectorXd>& sample_mu() const       { return sample_mu_; }
    /// Per-sample covariances Sigma_i, symmetrized, target-dimensional (empty if absent).
    const std::vector<Eigen::MatrixXd>& sample_Sigma() const    { return sample_Sigma_; }
    /// Batch index of each sample point.
    const std::vector<int>& point2batch() const                 { return point2batch_; }
    /// Sample index range [start, stop) of batch b.
    std::pair<int, int> batch_range( int b ) const
    {
        return std::make_pair(batch2point_start_.at(b), batch2point_stop_.at(b));
    }
    /// Vertex values of batch b (on the target mesh).
    const Eigen::VectorXd& batch_values( int b ) const          { return batch_values_.at(b); }
    /// Vertex moment fields over the source mesh (empty if absent); Sigma
    /// columns are vec(Sigma_v), target-dimensional.
    const Eigen::VectorXd& field_V() const                      { return field_V_; }
    const Eigen::MatrixXd& field_mu() const                     { return field_mu_; }
    const Eigen::MatrixXd& field_Sigma() const                  { return field_Sigma_; }
    /// Derived inverse-square-root field (column v = vec(Sigma_v^{-1/2})).
    const Eigen::MatrixXd& field_W() const                      { return field_W_; }

    /// Sets the vertex moment fields used to evaluate V(x), mu(x), Sigma(x)
    /// at query points: fields over the SOURCE mesh with TARGET-valued
    /// entries. Pass size-zero arrays for fields you do not have (validation
    /// will say if a configuration needs them). Shapes:
    /// V (num_source_vertices), mu (dim_target, num_source_vertices),
    /// Sigma (dim_target^2, num_source_vertices) with column v =
    /// vec(Sigma at vertex v).
    ///
    /// Sigma columns are symmetrized here and every vertex covariance must be
    /// strictly positive definite (throws listing the offending vertices
    /// otherwise, leaving the fields unchanged). Vertex-level validation is
    /// enough: lambda_min is concave, so every CG1-interpolated Sigma(x) is
    /// then SPD too. Repair noisy fields beforehand with clamp_spd_field —
    /// and note that a *near*-singular Sigma passes validation but amplifies
    /// through Sigma(x)^{-1/2} and det Sigma(x); see moments.hpp for how to
    /// choose the clamping floor. The validation eigendecompositions are
    /// kept as the inverse-square-root field W (column v =
    /// vec(Sigma_v^{-1/2})): whitened_affine and volume_det interpolate W
    /// directly, which is exact at the vertices, SPD everywhere by
    /// convexity, and avoids per-evaluation eigensolves.
    void set_moment_fields( const Eigen::Ref<const Eigen::VectorXd>& V,
                            const Eigen::Ref<const Eigen::MatrixXd>& mu,
                            const Eigen::Ref<const Eigen::MatrixXd>& Sigma )
    {
        const int nv = num_source_vertices();
        const int dt = dim_target_;
        if ( V.size() != 0 && V.size() != nv )
        {
            throw std::invalid_argument("psfi::ImpulseResponseField::set_moment_fields: V must have "
                                        "one entry per source vertex (or size zero)");
        }
        if ( mu.cols() != 0 && ( mu.rows() != dt || mu.cols() != nv ) )
        {
            throw std::invalid_argument("psfi::ImpulseResponseField::set_moment_fields: mu must have "
                                        "shape (dim_target, num_source_vertices) (or be empty)");
        }
        if ( Sigma.cols() != 0 && ( Sigma.rows() != dt * dt || Sigma.cols() != nv ) )
        {
            throw std::invalid_argument("psfi::ImpulseResponseField::set_moment_fields: Sigma must have "
                                        "shape (dim_target^2, num_source_vertices) (or be empty)");
        }

        // Symmetrize and validate the covariance field before assigning
        // anything, so a throw leaves the fields unchanged.
        Eigen::MatrixXd Sigma_sym = Sigma;
        Eigen::MatrixXd W_field(Sigma.rows(), Sigma.cols());
        if ( Sigma.cols() > 0 )
        {
            std::vector<std::pair<int, double>> bad; // (vertex, min eigenvalue)
            for ( int v = 0; v < nv; ++v )
            {
                const Eigen::Map<const Eigen::MatrixXd> S(Sigma.col(v).data(), dt, dt);
                const Eigen::MatrixXd S_sym = 0.5 * ( S + S.transpose() );
                Sigma_sym.col(v) = Eigen::Map<const Eigen::VectorXd>(S_sym.data(), dt * dt);
                Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(S_sym);
                if ( es.info() != Eigen::Success || es.eigenvalues().minCoeff() <= 0.0 )
                {
                    bad.emplace_back(v, es.eigenvalues().minCoeff());
                }
                else
                {
                    const Eigen::MatrixXd W = es.operatorInverseSqrt();
                    W_field.col(v) = Eigen::Map<const Eigen::VectorXd>(W.data(), dt * dt);
                }
            }
            if ( !bad.empty() )
            {
                std::ostringstream message;
                message << "psfi::ImpulseResponseField::set_moment_fields: Sigma is not positive "
                        << "definite at " << bad.size() << " vertex(es):";
                for ( size_t ii = 0; ii < bad.size() && ii < 10; ++ii )
                {
                    message << " v=" << bad[ii].first << " (min eig " << bad[ii].second << ")";
                }
                if ( bad.size() > 10 )
                {
                    message << " ... and " << ( bad.size() - 10 ) << " more";
                }
                message << ". Clean the field first — see psfi::clamp_spd_field; a floor of the "
                        << "local mesh spacing squared is a reasonable choice.";
                throw std::invalid_argument(message.str());
            }
        }
        field_V_     = V;
        field_mu_    = mu;
        field_Sigma_ = std::move(Sigma_sym);
        field_W_     = std::move(W_field);
    }

    /// Adds one impulse response batch: sample points (columns of `points`,
    /// arbitrary source-domain coordinates), the batch function as CG1
    /// vertex values `psi` on the target mesh, and optional per-sample
    /// moments — V_i scalar, mu_i and Sigma_i target-dimensional (pass
    /// size-zero arrays for moments you do not have). Which moments are
    /// supplied is fixed by the first batch; later batches must supply the
    /// same set. Sigma_i are symmetrized and must be positive definite
    /// (throws otherwise — sample moments come from a curated picking
    /// procedure and junk here is an error, unlike the vertex fields). With
    /// rebuild = false the kd-tree over sample points is left stale; call
    /// rebuild_kdtree() before predictions().
    void add_batch( const Eigen::Ref<const Eigen::MatrixXd>& points, // (dim_source, num_batch_points)
                    const Eigen::Ref<const Eigen::VectorXd>& psi,    // (num_target_vertices)
                    const Eigen::Ref<const Eigen::VectorXd>& V,      // (num_batch_points) or size 0
                    const Eigen::Ref<const Eigen::MatrixXd>& mu,     // (dim_target, num_batch_points) or empty
                    const std::vector<Eigen::MatrixXd>&      Sigma,  // num_batch_points entries or empty
                    bool rebuild = true )
    {
        const int nb = static_cast<int>(points.cols());
        const int dt = dim_target_;
        if ( points.rows() != dim_source_ || nb < 1 )
        {
            throw std::invalid_argument("psfi::ImpulseResponseField::add_batch: points must have shape "
                                        "(dim_source, num_batch_points) with at least one point");
        }
        if ( psi.size() != num_target_vertices() )
        {
            throw std::invalid_argument("psfi::ImpulseResponseField::add_batch: psi must have one entry "
                                        "per target-mesh vertex");
        }
        const bool have_V     = ( V.size() > 0 );
        const bool have_mu    = ( mu.cols() > 0 );
        const bool have_Sigma = !Sigma.empty();
        if ( have_V && V.size() != nb )
        {
            throw std::invalid_argument("psfi::ImpulseResponseField::add_batch: V must have one entry "
                                        "per batch point (or size zero)");
        }
        if ( have_mu && ( mu.rows() != dt || mu.cols() != nb ) )
        {
            throw std::invalid_argument("psfi::ImpulseResponseField::add_batch: mu must have shape "
                                        "(dim_target, num_batch_points) (or be empty)");
        }
        if ( have_Sigma && static_cast<int>(Sigma.size()) != nb )
        {
            throw std::invalid_argument("psfi::ImpulseResponseField::add_batch: Sigma must have one "
                                        "matrix per batch point (or be empty)");
        }
        if ( num_batches() == 0 )
        {
            has_sample_V_     = have_V;
            has_sample_mu_    = have_mu;
            has_sample_Sigma_ = have_Sigma;
        }
        else if ( have_V != has_sample_V_ || have_mu != has_sample_mu_ || have_Sigma != has_sample_Sigma_ )
        {
            throw std::invalid_argument("psfi::ImpulseResponseField::add_batch: all batches must supply "
                                        "the same set of per-sample moments (V/mu/Sigma) as the first batch");
        }

        // Validate and precompute Sigma-derived quantities before mutating
        // anything, so a throw leaves the field unchanged.
        std::vector<Eigen::MatrixXd> Sigma_sym, Sigma_inv, Sigma_sqrt;
        std::vector<double>          Sigma_det;
        if ( have_Sigma )
        {
            Sigma_sym.reserve(nb); Sigma_inv.reserve(nb); Sigma_sqrt.reserve(nb); Sigma_det.reserve(nb);
            for ( int ii = 0; ii < nb; ++ii )
            {
                if ( Sigma[ii].rows() != dt || Sigma[ii].cols() != dt )
                {
                    throw std::invalid_argument("psfi::ImpulseResponseField::add_batch: Sigma["
                                                + std::to_string(ii) + "] must be "
                                                "(dim_target, dim_target)");
                }
                Eigen::MatrixXd S = 0.5 * ( Sigma[ii] + Sigma[ii].transpose() );
                Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(S);
                if ( es.info() != Eigen::Success || es.eigenvalues().minCoeff() <= 0.0 )
                {
                    throw std::invalid_argument("psfi::ImpulseResponseField::add_batch: Sigma["
                                                + std::to_string(ii) + "] is not positive definite "
                                                "(clean sample covariances first; see psfi::clamp_spd)");
                }
                Sigma_sym .push_back(std::move(S));
                Sigma_inv .push_back(es.operatorInverseSqrt() * es.operatorInverseSqrt());
                Sigma_sqrt.push_back(es.operatorSqrt());
                Sigma_det .push_back(es.eigenvalues().prod());
            }
        }

        const int batch_ind = num_batches();
        batch2point_start_.push_back(num_sample_points());
        for ( int ii = 0; ii < nb; ++ii )
        {
            point2batch_.push_back(batch_ind);
            sample_points_.push_back(points.col(ii));
            if ( have_V )
            {
                sample_V_.push_back(V(ii));
            }
            if ( have_mu )
            {
                sample_mu_.push_back(mu.col(ii));
            }
            if ( have_Sigma )
            {
                sample_Sigma_     .push_back(std::move(Sigma_sym [ii]));
                sample_Sigma_inv_ .push_back(std::move(Sigma_inv [ii]));
                sample_Sigma_sqrt_.push_back(std::move(Sigma_sqrt[ii]));
                sample_Sigma_det_ .push_back(Sigma_det[ii]);
            }
        }
        batch2point_stop_.push_back(num_sample_points());
        batch_values_.push_back(psi);

        kdtree_dirty_ = true;
        if ( rebuild )
        {
            rebuild_kdtree();
        }
    }

    /// (Re)builds the kd-tree over the sample points (needed after add_batch
    /// calls made with rebuild = false).
    void rebuild_kdtree()
    {
        if ( num_sample_points() > 0 )
        {
            kdtree_.build(sample_points());
        }
        kdtree_dirty_ = false;
    }

    /// Throws std::invalid_argument listing every piece of data the
    /// configuration needs but the field does not have (and any invalid
    /// parameter values); returns normally iff predictions(y, x, config) can
    /// run. The requirements per axis (N = batches_normalized):
    ///
    ///   frame  translation      : dim_source == dim_target
    ///   frame  mean_translation : sample mu, field mu
    ///   frame  whitened_affine  : sample mu + Sigma, field mu + Sigma
    ///   scaling none            : sample V if N
    ///   scaling volume          : field V, plus sample V if !N
    ///   scaling volume_det      : field V + Sigma, sample Sigma, plus sample V if !N
    ///   support ellipsoid       : sample mu + Sigma, tau > 0
    void validate( const EvalConfig& config ) const
    {
        std::vector<std::string> missing;
        auto need = [&]( bool ok, const char* what )
        {
            if ( !ok )
            {
                missing.push_back(what);
            }
        };

        if ( config.num_neighbors < 1 )
        {
            missing.push_back("num_neighbors >= 1");
        }
        if ( config.frame == Frame::translation && dim_source_ != dim_target_ )
        {
            missing.push_back("equal source and target dimensions (frame translation forms y - x "
                              "+ x_i; use mean_translation for cross-dimension kernels)");
        }
        if ( config.frame == Frame::mean_translation || config.frame == Frame::whitened_affine )
        {
            need(has_sample_mu_, "per-sample mu (add_batch)");
            need(has_field_mu(), "vertex field mu (set_moment_fields)");
        }
        if ( config.frame == Frame::whitened_affine )
        {
            need(has_sample_Sigma_, "per-sample Sigma (add_batch)");
            need(has_field_Sigma(), "vertex field Sigma (set_moment_fields)");
        }
        if ( config.scaling == Scaling::none && batches_normalized_ )
        {
            need(has_sample_V_, "per-sample V (add_batch; needed to un-normalize batches)");
        }
        if ( config.scaling == Scaling::volume || config.scaling == Scaling::volume_det )
        {
            need(has_field_V(), "vertex field V (set_moment_fields)");
            if ( !batches_normalized_ )
            {
                need(has_sample_V_, "per-sample V (add_batch)");
            }
        }
        if ( config.scaling == Scaling::volume_det )
        {
            need(has_sample_Sigma_, "per-sample Sigma (add_batch)");
            need(has_field_Sigma(), "vertex field Sigma (set_moment_fields)");
        }
        if ( config.support == Support::ellipsoid )
        {
            need(has_sample_mu_, "per-sample mu (add_batch)");
            need(has_sample_Sigma_, "per-sample Sigma (add_batch)");
            if ( !( config.tau > 0.0 ) )
            {
                missing.push_back("tau > 0");
            }
        }

        if ( !missing.empty() )
        {
            std::string message = "psfi::ImpulseResponseField::validate: this configuration requires:";
            for ( const std::string& mm : missing )
            {
                message += "\n  - " + mm;
            }
            throw std::invalid_argument(message);
        }
    }

    /// Per-neighbor kernel predictions at the pair (y, x) — y in the target
    /// domain, x in the source domain, both arbitrary coordinates: for each
    /// of the (up to) num_neighbors sample points x_i nearest to x, the pair
    /// (x_i, f_i) with f_i = s_i * phi_i(T_i(y)) per the configuration. The
    /// support gate is tested before any mesh lookup: gated samples
    /// contribute f_i = 0 at negligible cost wherever T_i(y) lies (the
    /// far-field short circuit), while ungated samples whose T_i(y) falls
    /// outside the target mesh are excluded. Returns no predictions when x
    /// lies outside the source mesh but the configuration needs moment
    /// fields at x: the kernel is uninformed there and evaluates to zero.
    /// Thread-safe (const; no caches).
    std::vector<Prediction> predictions( const Eigen::Ref<const Eigen::VectorXd>& y,
                                         const Eigen::Ref<const Eigen::VectorXd>& x,
                                         const EvalConfig& config ) const
    {
        if ( y.size() != dim_target_ || x.size() != dim_source_ )
        {
            throw std::invalid_argument("psfi::ImpulseResponseField::predictions: y must have length "
                                        "dim_target and x length dim_source");
        }
        if ( num_sample_points() == 0 )
        {
            return {};
        }
        validate(config);
        if ( kdtree_dirty_ )
        {
            throw std::logic_error("psfi::ImpulseResponseField::predictions: kd-tree is stale; call "
                                   "rebuild_kdtree() after add_batch(..., rebuild=false)");
        }

        // Query-side quantities (CG1 interpolation of the source-mesh
        // vertex fields at x; all values are target-dimensional).
        const bool need_mu_x = ( config.frame == Frame::mean_translation
                                 || config.frame == Frame::whitened_affine );
        const bool need_W_x  = ( config.frame == Frame::whitened_affine
                                 || config.scaling == Scaling::volume_det );
        const bool need_V_x  = ( config.scaling == Scaling::volume
                                 || config.scaling == Scaling::volume_det );

        const int dt = dim_target_;
        double          V_x = 0.0;
        double          det_Sigma_x = 0.0;
        Eigen::VectorXd mu_x;
        Eigen::MatrixXd W_x; // interpolated inverse-square-root field (SPD by convexity)
        if ( need_mu_x || need_W_x || need_V_x )
        {
            int cell;
            Eigen::VectorXd alpha;
            if ( !locate(source_mesh(), x, cell, alpha) )
            {
                return {}; // x outside the source domain: kernel is zero there
            }
            const Eigen::MatrixXi& scells = source_mesh().cells();
            if ( need_V_x )
            {
                V_x = 0.0;
                for ( int kk = 0; kk < dim_source_ + 1; ++kk )
                {
                    V_x += alpha(kk) * field_V_(scells(kk, cell));
                }
            }
            if ( need_mu_x )
            {
                mu_x = Eigen::VectorXd::Zero(dt);
                for ( int kk = 0; kk < dim_source_ + 1; ++kk )
                {
                    mu_x += alpha(kk) * field_mu_.col(scells(kk, cell));
                }
            }
            if ( need_W_x )
            {
                W_x = Eigen::MatrixXd::Zero(dt, dt);
                for ( int kk = 0; kk < dim_source_ + 1; ++kk )
                {
                    W_x += alpha(kk)
                        * Eigen::Map<const Eigen::MatrixXd>(
                              field_W_.col(scells(kk, cell)).data(), dt, dt);
                }
                // det Sigma(x) := det(W(x))^{-2}, consistent with the W-field
                // definition of the frame map (W is SPD, so det W > 0).
                const double det_W = W_x.determinant();
                det_Sigma_x = 1.0 / ( det_W * det_W );
            }
        }

        // For whitened_affine the whitened offset is shared by all neighbors,
        // and |W(x)(y - mu(x))| is every neighbor's Mahalanobis distance:
        // one test gates the entire neighbor set.
        Eigen::VectorXd wvec;
        bool shared_gate_inside = true;
        if ( config.frame == Frame::whitened_affine )
        {
            wvec = W_x * ( y - mu_x );
            shared_gate_inside = ( wvec.squaredNorm() <= config.tau * config.tau );
        }

        // k nearest sample points to x.
        const int k_eff = std::min(config.num_neighbors, num_sample_points());
        Eigen::MatrixXd xq(dim_source_, 1);
        xq.col(0) = x;
        const Eigen::MatrixXi nearest = kdtree_.query(xq, k_eff, 1).first;

        std::vector<Prediction> out;
        out.reserve(k_eff);
        for ( int jj = 0; jj < k_eff; ++jj )
        {
            const int ii = nearest(jj, 0);
            const Eigen::VectorXd& xi = sample_points_[ii];

            Eigen::VectorXd z;
            switch ( config.frame )
            {
                case Frame::identity:         z = y;                          break;
                case Frame::translation:      z = y - x + xi;                 break;
                case Frame::mean_translation: z = y - mu_x + sample_mu_[ii];  break;
                case Frame::whitened_affine:  z = sample_mu_[ii] + sample_Sigma_sqrt_[ii] * wvec; break;
            }

            // Support gate first — the far-field short circuit: gated
            // predictions are pushed with value 0 without any mesh lookup.
            if ( config.support == Support::ellipsoid )
            {
                bool inside = shared_gate_inside;
                if ( config.frame != Frame::whitened_affine )
                {
                    const Eigen::VectorXd dz = z - sample_mu_[ii];
                    inside = ( dz.dot(sample_Sigma_inv_[ii] * dz) <= config.tau * config.tau );
                }
                if ( !inside )
                {
                    out.push_back(Prediction{ii, xi, 0.0});
                    continue;
                }
            }

            int cell;
            Eigen::VectorXd alpha;
            if ( !locate(mesh_target_, z, cell, alpha) )
            {
                continue; // ungated point outside the target domain: sample excluded
            }

            const Eigen::VectorXd& psi = batch_values_[point2batch_[ii]];
            double raw = 0.0;
            for ( int kk = 0; kk < dt + 1; ++kk )
            {
                raw += alpha(kk) * psi(mesh_target_.cells()(kk, cell));
            }

            double s = 1.0;
            switch ( config.scaling )
            {
                case Scaling::none:
                    s = batches_normalized_ ? sample_V_[ii] : 1.0;
                    break;
                case Scaling::volume:
                    s = batches_normalized_ ? V_x : V_x / sample_V_[ii];
                    break;
                case Scaling::volume_det:
                    s = ( batches_normalized_ ? V_x : V_x / sample_V_[ii] )
                        * std::sqrt(sample_Sigma_det_[ii] / det_Sigma_x);
                    break;
            }
            out.push_back(Prediction{ii, xi, s * raw});
        }
        return out;
    }

private:
    /// The domain-membership test: currently "inside the (local) mesh". Kept
    /// as the single hook where a global-domain indicator would plug in for
    /// distributed use. Outputs the containing cell and its barycentric
    /// coordinates when inside.
    static bool locate( const etree::SimplexMesh& mesh,
                        const Eigen::Ref<const Eigen::VectorXd>& p,
                        int& cell, Eigen::VectorXd& alpha )
    {
        cell = mesh.cell_tree().first_collision(p);
        if ( cell < 0 )
        {
            return false;
        }
        alpha = mesh.cell_tree().affine_coordinates(cell, p);
        return true;
    }

    int dim_source_ = 0;
    int dim_target_ = 0;
    etree::SimplexMesh                mesh_target_;
    std::optional<etree::SimplexMesh> mesh_source_; // absent: source = target
    bool batches_normalized_ = true;

    bool has_sample_V_     = false;
    bool has_sample_mu_    = false;
    bool has_sample_Sigma_ = false;

    std::vector<Eigen::VectorXd> sample_points_;    // source-domain coordinates
    std::vector<double>          sample_V_;
    std::vector<Eigen::VectorXd> sample_mu_;        // target-domain
    std::vector<Eigen::MatrixXd> sample_Sigma_;     // target-dimensional, symmetrized
    std::vector<Eigen::MatrixXd> sample_Sigma_inv_; // derived at add_batch
    std::vector<Eigen::MatrixXd> sample_Sigma_sqrt_;// symmetric PSD square root
    std::vector<double>          sample_Sigma_det_;

    std::vector<int>             point2batch_;
    std::vector<int>             batch2point_start_;
    std::vector<int>             batch2point_stop_;
    std::vector<Eigen::VectorXd> batch_values_; // CG1 vertex vectors on the target mesh

    Eigen::VectorXd field_V_;     // (num_source_vertices) or empty
    Eigen::MatrixXd field_mu_;    // (dim_target, num_source_vertices) or empty
    Eigen::MatrixXd field_Sigma_; // (dim_target^2, num_source_vertices), col v = vec(Sigma_v), or empty
    Eigen::MatrixXd field_W_;     // derived: column v = vec(Sigma_v^{-1/2}), or empty

    etree::KDTree kdtree_; // over the sample points (source domain)
    bool          kdtree_dirty_ = false;
};

} // end namespace psfi

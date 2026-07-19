// Frog kernel: PSF interpolation end to end
//
// The "frog" kernel (eq. 7.4 of the localpsf paper) is a spatially varying
// blur on the unit square: the impulse response at x is an anisotropic bumpy
// blob that rotates with theta(x) = (pi/2)(x_1 + x_2), is modulated so the
// kernel is mildly negative (a = 1), and is scaled by the boundary bump
// V(x) = x_1(1-x_1) x_2(1-x_2). Because the kernel is analytic, applying
// the operator to a weighted Dirac comb is the same as summing scaled kernel
// columns, and the moments are known a priori by formula: V(x) exactly
// (the modulation integrates to zero against the Gaussian), mu(x) = x and
// Sigma(x) = R(theta)^T Sigma_0 R(theta) up to O(e^{-4}) corrections that we
// deliberately ignore — a-priori moments are never perfect.
//
// The pipeline: support ellipsoids from the moment formulas -> greedy
// non-overlapping batch picking (etree) -> impulse response batches ->
// kernel evaluation anywhere by transported-impulse interpolation (psfi).
// The figures walk through it: the first five batches with their support
// ellipsoids and the two target points (red); the sample points, with the
// ones actually used to interpolate at the targets in black and the rest in
// gray; true vs reconstructed point spread functions at the targets with 1
// and 5 batches; and maps of the relative column error
// ||Phi(.,x) - Phi~(.,x)|| / ||Phi(.,x)|| over x for 1/5/10 batches
// (cf. Figure 6 of the paper), for the paper's configuration
// (mean_translation + volume) and for whitened_affine + volume_det, which
// additionally deforms each impulse to the local ellipsoid shape — on this
// rotating kernel that pays off, as the printed error quantiles show. Each
// error map is shown with k = 10 neighbors and with k = 1 (nearest impulse
// only). Denominators are floored at 1% of the largest column norm so the
// vanishing-V boundary ring reads as zero rather than 0/0; errors are
// clipped at 1 for plotting (shared color scale 0..1).
//
// The 5-batch k = 10 maps show a localized hot spot near (0.2, 0.75). This
// is an interference effect of the sparse-sampling regime, not a coverage
// hole: with ~30 samples the 10 nearest neighbors include impulses
// misrotated by up to ~45 degrees, each individually 20-60% wrong, and the
// interpolant — a signed linear combination of their predictions — usually
// cancels those individual errors (which is why k = 10 beats k = 1 almost
// everywhere) but happens to add them constructively in that region, as the
// clean k = 1 maps there confirm. More batches shrink the neighbor radius
// and the effect disappears by 10 batches.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <memory>
#include <set>
#include <vector>

#include "etree/etree.hpp"
#include "etree/plot2d.hpp"
#include "psfi/psfi.hpp"

using namespace etree;
using namespace psfi;

namespace {

const double kA = 1.0;                  // modulation strength (mildly negative kernel)
const double kVol = 1.0;
const Eigen::Matrix2d kSigma0 = Eigen::Vector2d(0.01, 0.0025).asDiagonal();

Eigen::Matrix2d rotation( const Eigen::Vector2d& x )
{
    const double theta = 0.5 * M_PI * ( x(0) + x(1) );
    Eigen::Matrix2d R;
    R << std::cos(theta), -std::sin(theta),
         std::sin(theta),  std::cos(theta);
    return R;
}

double bump( const Eigen::Vector2d& x )
{
    return x(0) * ( 1.0 - x(0) ) * x(1) * ( 1.0 - x(1) );
}

// The frog kernel Phi(y, x) = phi_x(y).
double frog( const Eigen::Vector2d& y, const Eigen::Vector2d& x )
{
    const Eigen::Vector2d p = rotation(x) * ( y - x );
    const double maha2 = p(0) * p(0) / kSigma0(0, 0) + p(1) * p(1) / kSigma0(1, 1);
    const double G = kVol / ( 2.0 * M_PI * std::sqrt(kSigma0.determinant()) )
        * std::exp(-0.5 * maha2);
    const double modulation = std::cos(p(0) / ( std::sqrt(kSigma0(0, 0)) / 2.0 ))
        * std::sin(p(1) / ( std::sqrt(kSigma0(1, 1)) / 2.0 ));
    return bump(x) * ( 1.0 + kA * modulation ) * G;
}

// A-priori moments (exact up to the ignored modulation corrections).
double V_of( const Eigen::Vector2d& x )          { return bump(x) * kVol; }
Eigen::Matrix2d Sigma_of( const Eigen::Vector2d& x )
{
    const Eigen::Matrix2d R = rotation(x);
    return R.transpose() * kSigma0 * R;
}

double percentile( std::vector<double> v, double q )
{
    std::sort(v.begin(), v.end());
    return v[static_cast<size_t>(q * ( v.size() - 1 ))];
}

} // end anonymous namespace

int main()
{
    // ---- Mesh: structured triangulation of [0, 1]^2 ----
    const int m = 40;
    const int nv = ( m + 1 ) * ( m + 1 );
    Eigen::MatrixXd vertices(2, nv);
    for ( int jj = 0; jj <= m; ++jj )
    {
        for ( int ii = 0; ii <= m; ++ii )
        {
            vertices.col(jj * ( m + 1 ) + ii) = Eigen::Vector2d(ii / double(m), jj / double(m));
        }
    }
    Eigen::MatrixXi cells(3, 2 * m * m);
    int cc = 0;
    for ( int jj = 0; jj < m; ++jj )
    {
        for ( int ii = 0; ii < m; ++ii )
        {
            const int v00 = jj * ( m + 1 ) + ii;
            const int v10 = v00 + 1;
            const int v01 = v00 + ( m + 1 );
            const int v11 = v01 + 1;
            cells.col(cc++) = Eigen::Vector3i(v00, v10, v11);
            cells.col(cc++) = Eigen::Vector3i(v00, v11, v01);
        }
    }

    // ---- Impulse response field with a-priori moment fields ----
    auto F = std::make_shared<ImpulseResponseField>(vertices, cells);
    {
        Eigen::VectorXd field_V(nv);
        Eigen::MatrixXd field_mu = vertices;
        Eigen::MatrixXd field_Sigma(4, nv);
        for ( int v = 0; v < nv; ++v )
        {
            field_V(v) = V_of(vertices.col(v));
            const Eigen::Matrix2d S = Sigma_of(vertices.col(v));
            field_Sigma.col(v) = Eigen::Map<const Eigen::VectorXd>(S.data(), 4);
        }
        // The Sigma formula is SPD everywhere, but V vanishes on the
        // boundary; that is fine for the *field* (it multiplies), only the
        // sample candidates below need V > 0.
        F->set_moment_fields(field_V, field_mu, field_Sigma);
    }

    // ---- Candidate sample points and their support ellipsoids ----
    const double tau = 3.0;
    std::vector<int> candidates;
    double V_max = 0.0;
    for ( int v = 0; v < nv; ++v )
    {
        V_max = std::max(V_max, V_of(vertices.col(v)));
    }
    for ( int v = 0; v < nv; ++v )
    {
        if ( V_of(vertices.col(v)) > 1e-5 * V_max ) // paper's epsilon_V threshold
        {
            candidates.push_back(v);
        }
    }
    std::vector<Ellipsoid> ellipsoids(candidates.size());
    Eigen::MatrixXd anchors(2, candidates.size());
    for ( size_t ii = 0; ii < candidates.size(); ++ii )
    {
        const Eigen::Vector2d x = vertices.col(candidates[ii]);
        ellipsoids[ii] = Ellipsoid{x, Sigma_of(x)};
        anchors.col(ii) = x;
    }

    // ---- Greedy non-overlapping batch picking (etree) ----
    const int num_batches = 10;
    EllipsoidTree etree_tree(ellipsoids, tau);
    const std::vector<std::vector<int>> batches =
        pick_ellipsoid_batches(etree_tree, anchors, num_batches);

    const std::vector<Eigen::Vector2d> targets = {{0.62, 0.33}, {0.28, 0.71}};
    const Style ellipse_style{with_alpha(colors::black(), 0.65), 1.0, colors::transparent()};
    const Style target_style{colors::vermillion(), 1.0, colors::vermillion()};

    // ---- True kernel matrix and column norms (for PSF and error figures) ----
    Eigen::MatrixXd T(nv, nv);
    for ( int jj = 0; jj < nv; ++jj )
    {
        for ( int ii = 0; ii < nv; ++ii )
        {
            T(ii, jj) = frog(vertices.col(ii), vertices.col(jj));
        }
    }
    const Eigen::VectorXd T_col_norms = T.colwise().norm();
    const double norm_floor = 1e-2 * T_col_norms.maxCoeff();

    // True PSFs at the targets (color scales shared with the reconstructions).
    std::vector<Eigen::VectorXd> true_psf(targets.size());
    for ( size_t tt = 0; tt < targets.size(); ++tt )
    {
        true_psf[tt].resize(nv);
        for ( int v = 0; v < nv; ++v )
        {
            true_psf[tt](v) = frog(vertices.col(v), targets[tt]);
        }
        Plot2D fig;
        FieldOptions opts;
        draw_cg1_field(fig, F->mesh(), true_psf[tt], opts);
        fig.add_marker(targets[tt], 4.0, target_style);
        char path[64];
        std::snprintf(path, sizeof(path), "%02zu_true_psf_target%zu.png", 7 + 3 * tt, tt + 1);
        fig.save_png(path, 560);
    }

    // ---- Evaluation configurations ----
    EvalConfig cfg_paper; // mean_translation + volume: the paper's method
    cfg_paper.tau = tau;
    cfg_paper.num_neighbors = 10;
    EvalConfig cfg_whitened = cfg_paper; // + local ellipsoid deformation
    cfg_whitened.frame = Frame::whitened_affine;
    cfg_whitened.scaling = Scaling::volume_det;
    const RBFScheme rbf; // gaussian, C_RBF = 3, linear tail

    const std::vector<std::pair<const char*, EvalConfig>> configs = {
        {"mean_translation+volume  ", cfg_paper},
        {"whitened_affine+volume_det", cfg_whitened}};

    std::printf("frog kernel on a %d x %d mesh (%d vertices), %zu sample candidates\n",
                m, m, nv, candidates.size());
    std::printf("tau = %.1f, k = %d neighbors, gaussian RBF with C_RBF = %.1f\n\n",
                tau, cfg_paper.num_neighbors, rbf.shape);

    // ---- Add batches; at 1, 5, and 10 batches evaluate everything ----
    const std::set<int> stages = {1, 5, 10};
    for ( int b = 0; b < num_batches; ++b )
    {
        const int nb = static_cast<int>(batches[b].size());
        Eigen::MatrixXd points(2, nb);
        Eigen::VectorXd V(nb);
        Eigen::MatrixXd mu(2, nb);
        std::vector<Eigen::MatrixXd> Sigma(nb);
        for ( int ii = 0; ii < nb; ++ii )
        {
            const Eigen::Vector2d x = vertices.col(candidates[batches[b][ii]]);
            points.col(ii) = x;
            V(ii) = V_of(x);
            mu.col(ii) = x;
            Sigma[ii] = Sigma_of(x);
        }
        Eigen::VectorXd psi = Eigen::VectorXd::Zero(nv);
        for ( int v = 0; v < nv; ++v )
        {
            for ( int ii = 0; ii < nb; ++ii )
            {
                psi(v) += frog(vertices.col(v), points.col(ii)) / V(ii);
            }
        }
        F->add_batch(points, psi, V, mu, Sigma);
        std::printf("batch %2d: %2d impulse responses\n", b + 1, nb);

        // Batch figure (first five batches): the batch function, its support
        // ellipsoids, and the targets.
        if ( b < 5 )
        {
            Plot2D fig;
            draw_cg1_field(fig, F->mesh(), psi);
            for ( int ii = 0; ii < nb; ++ii )
            {
                fig.add(Ellipsoid{mu.col(ii), Sigma[ii]}, tau, ellipse_style);
            }
            for ( const Eigen::Vector2d& t : targets )
            {
                fig.add_marker(t, 4.0, target_style);
            }
            char path[64];
            std::snprintf(path, sizeof(path), "%02d_batch_%d.png", b + 1, b + 1);
            fig.save_png(path, 560);
        }

        if ( stages.count(b + 1) == 0 )
        {
            continue;
        }
        const int stage = b + 1;

        // Reconstructed PSFs at the targets (paper configuration, stages 1 and 5).
        if ( stage <= 5 )
        {
            KernelEvaluator K(F, nullptr, cfg_paper, rbf);
            Eigen::MatrixXd targets_mat(2, targets.size());
            for ( size_t tt = 0; tt < targets.size(); ++tt )
            {
                targets_mat.col(tt) = targets[tt];
            }
            const Eigen::MatrixXd psf = K.block(vertices, targets_mat);
            for ( size_t tt = 0; tt < targets.size(); ++tt )
            {
                Plot2D fig;
                FieldOptions opts;
                opts.vmin = true_psf[tt].minCoeff(); // shared scale with the true PSF
                opts.vmax = true_psf[tt].maxCoeff();
                draw_cg1_field(fig, F->mesh(), psf.col(tt), opts);
                fig.add_marker(targets[tt], 4.0, target_style);
                char path[64];
                std::snprintf(path, sizeof(path), "%02zu_psf_target%zu_batches%d.png",
                              7 + 3 * tt + ( stage == 1 ? 1 : 2 ), tt + 1, stage);
                fig.save_png(path, 560);
            }
        }

        // Column-error maps (cf. paper Figure 6), both configurations, with
        // the full k = 10 neighbor pooling and with k = 1 (nearest impulse only).
        for ( size_t qq = 0; qq < configs.size(); ++qq )
        {
            for ( int num_neighbors : {1, 10} )
            {
                EvalConfig cfg_k = configs[qq].second;
                cfg_k.num_neighbors = num_neighbors;
                KernelEvaluator K(F, nullptr, cfg_k, rbf);
                const Eigen::MatrixXd B = K.block(vertices, vertices);
                Eigen::VectorXd err(nv);
                std::vector<double> informative_errs;
                for ( int jj = 0; jj < nv; ++jj )
                {
                    const double e = ( B.col(jj) - T.col(jj) ).norm()
                        / std::max(T_col_norms(jj), norm_floor);
                    err(jj) = std::min(e, 1.0);
                    if ( T_col_norms(jj) >= norm_floor )
                    {
                        informative_errs.push_back(e);
                    }
                }
                std::printf("  %s  k=%2d  batches=%2d  median rel col err %.3f, p90 %.3f\n",
                            configs[qq].first, num_neighbors, stage,
                            percentile(informative_errs, 0.5), percentile(informative_errs, 0.9));

                Plot2D fig;
                FieldOptions opts;
                opts.vmin = 0.0;
                opts.vmax = 1.0;
                draw_cg1_field(fig, F->mesh(), err, opts);
                char path[64];
                std::snprintf(path, sizeof(path), "%2zu_error_%s_batches%02d_k%02d.png",
                              13 + qq, ( qq == 0 ? "paper" : "whitened" ), stage, num_neighbors);
                fig.save_png(path, 560);
            }
        }
    }

    // ---- Sample points: used for interpolation at the targets vs not ----
    {
        std::set<int> used;
        for ( const Eigen::Vector2d& t : targets )
        {
            for ( const Prediction& p : F->predictions(t, t, cfg_paper) )
            {
                used.insert(p.sample_index);
            }
        }
        Plot2D fig;
        fig.add(Box{Eigen::Vector2d(0, 0), Eigen::Vector2d(1, 1)},
                Style{with_alpha(colors::black(), 0.4), 1.0, colors::transparent()});
        const Eigen::MatrixXd P = F->sample_points();
        for ( int ii = 0; ii < P.cols(); ++ii )
        {
            const bool is_used = ( used.count(ii) > 0 );
            const Style s = is_used ? Style{colors::black(), 1.0, colors::black()}
                                    : Style{colors::gray(), 1.0, colors::gray()};
            fig.add_marker(P.col(ii), is_used ? 3.5 : 2.5, s);
        }
        for ( const Eigen::Vector2d& t : targets )
        {
            fig.add_marker(t, 4.5, target_style);
        }
        fig.save_svg("06_sample_points.svg", 560);
        std::printf("\n%d sample points total; %zu used to interpolate at the %zu targets\n",
                    F->num_sample_points(), used.size(), targets.size());
    }

    return 0;
}

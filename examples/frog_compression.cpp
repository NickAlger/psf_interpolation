// Frog kernel compression: dense, block low rank, global low rank
//
// The downstream half of the pipeline: the interpolated frog kernel of the
// [frog kernel example](frog_kernel.md) is turned into compressed matrix
// formats. Sources and targets are both the mesh vertices, so the object
// being compressed is the (num_vertices x num_vertices) kernel matrix of
// the psfi evaluator itself — the reference everything is compared against.
// (If the impulse batches came from forward applies of an operator A, the
// compressed matrix's `apply` approximates A in the nodal kernel sense and
// `applyT` approximates A^T; mass matrices stay with the consumer,
// A = M Phi M.)
//
// The block low rank (BRLR) format partitions the SOURCE axis into
// spatially coherent subdomains (recursive median bisection); each block
// couples its sources to the target set where its kernel columns can be
// nonzero, computed exactly from the support-ellipsoid gate — entries
// outside are exactly zero, so the block structure is lossless and all
// error comes from per-block truncation. The figures show the partition
// with one block's target set and support ellipsoids; the per-block ranks
// at the headline tolerance as a map over the source domain; and relative
// column-error maps comparing BRLR against a global low rank (GLR) of THE
// SAME total storage — the locality the block format keeps and a single
// global factorization loses. The printed table sweeps the tolerance:
// storage and error for BRLR vs storage-matched GLR, plus the GLR obtained
// from the BRLR by randomized SVD of its applies alone (the cheap route at
// scale: no further kernel evaluations), which matches the directly
// truncated GLR closely.
//
// Column errors are relative per column with denominators floored at 1% of
// the largest column norm (the vanishing-V boundary ring reads as zero
// rather than 0/0) and clipped at 1 for plotting; both error maps share
// the 0..1 color scale. Per-block compression here uses the dense-SVD
// reference path, which keeps the printed numbers toolchain-stable for the
// documentation freshness check; ACA+ — the production path at scale,
// whose sampled pivot chains are deterministic per seed but not across
// compilers — is exercised in the test suite instead.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <memory>
#include <vector>

#include "etree/etree.hpp"
#include "etree/plot2d.hpp"
#include "psfi/psfi.hpp"

using namespace etree;
using namespace psfi;

namespace {

const double kA = 1.0;
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

// The frog kernel Phi(y, x) = phi_x(y) (eq. 7.4 of the paper).
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

double V_of( const Eigen::Vector2d& x )          { return bump(x) * kVol; }
Eigen::Matrix2d Sigma_of( const Eigen::Vector2d& x )
{
    const Eigen::Matrix2d R = rotation(x);
    return R.transpose() * kSigma0 * R;
}

int median_int( std::vector<int> v )
{
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

// Vertical viridis colorbar just right of the unit square, labeled at the
// bottom / middle / top values.
void add_colorbar( Plot2D& fig, double vmin, double vmax, const char* fmt )
{
    const double x0 = 1.04;
    const double x1 = 1.08;
    const int nseg = 64;
    for ( int ii = 0; ii < nseg; ++ii )
    {
        const double t0 = ii / double(nseg);
        const double t1 = ( ii + 1 ) / double(nseg);
        fig.add(Box{Eigen::Vector2d(x0, t0), Eigen::Vector2d(x1, t1 + 0.002)},
                Style{colors::transparent(), 0.0, colormap_viridis(( t0 + t1 ) / 2.0)});
    }
    fig.add(Box{Eigen::Vector2d(x0, 0.0), Eigen::Vector2d(x1, 1.0)},
            Style{with_alpha(colors::black(), 0.8), 1.0, colors::transparent()});
    for ( double t : { 0.0, 0.5, 1.0 } )
    {
        char label[32];
        std::snprintf(label, sizeof(label), fmt, vmin + t * ( vmax - vmin ));
        fig.add_text(Eigen::Vector2d(x1 + 0.015, t - 0.01), label, 12.0, colors::black(),
                     TextAnchor::start);
    }
}

// The docs generator emits <stem>.caption.md above the matching figure.
void write_caption( const char* figure_stem, const std::string& text )
{
    char path[128];
    std::snprintf(path, sizeof(path), "%s.caption.md", figure_stem);
    std::FILE* f = std::fopen(path, "w");
    if ( f )
    {
        std::fputs(text.c_str(), f);
        std::fclose(f);
    }
}

} // end anonymous namespace

int main()
{
    // ---- Mesh, a-priori moment fields, impulse batches (as in frog_kernel) ----
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
        F->set_moment_fields(field_V, field_mu, field_Sigma);
    }

    const double tau = 3.0;
    std::vector<int> candidates;
    double V_max = 0.0;
    for ( int v = 0; v < nv; ++v )
    {
        V_max = std::max(V_max, V_of(vertices.col(v)));
    }
    for ( int v = 0; v < nv; ++v )
    {
        if ( V_of(vertices.col(v)) > 1e-5 * V_max )
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
    EllipsoidTree etree_tree(ellipsoids, tau);
    const int num_batches = 10;
    const std::vector<std::vector<int>> batches =
        pick_ellipsoid_batches(etree_tree, anchors, num_batches);
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
    }

    // ---- The kernel to compress: whitened_affine + volume_det, gated ----
    EvalConfig cfg;
    cfg.frame         = Frame::whitened_affine;
    cfg.scaling       = Scaling::volume_det;
    cfg.tau           = tau;
    cfg.num_neighbors = 10;
    const KernelEvaluator K(F, nullptr, cfg);

    std::printf("frog kernel on a %d x %d mesh: kernel matrix is %d x %d (%.2g dense entries)\n",
                m, m, nv, nv, double(nv) * nv);

    // Reference: the dense kernel matrix (this IS pipeline step "dense").
    const Eigen::MatrixXd A = K.block(vertices, vertices);
    const double norm_A = A.norm();
    const Eigen::VectorXd col_norms = A.colwise().norm();
    const double norm_floor = 1e-2 * col_norms.maxCoeff();

    // ---- Source partition ----
    const int cutoff = 256; // ~8 subdomains of ~210 sources, a few ellipsoid widths wide
    const std::vector<std::vector<int>> partition =
        recursive_bisection_partition(vertices, cutoff);
    std::printf("source partition: %zu subdomains, cutoff %d\n\n", partition.size(), cutoff);

    // ---- Tolerance sweep: BRLR vs storage-matched GLR ----
    const double headline_rtol = 1e-2;
    KernelLowRankOptions options;
    options.method = CompressionMethod::dense_svd; // see the intro note
    BlockLowRank B_headline;
    std::vector<int> ranks_headline;
    std::printf("            BRLR                     GLR at equal storage      GLR from BRLR applies\n");
    std::printf("  rtol      storage   rel err        rank   rel err            rel err\n");
    for ( double rtol : { 1e-1, 1e-2, 1e-3, 1e-4 } )
    {
        const BlockLowRankBuildResult r =
            block_low_rank(K, vertices, vertices, partition, rtol, options);
        const double err = ( A - r.matrix.to_dense() ).norm() / norm_A;
        const long long storage = r.matrix.storage_entries();

        // A single global factorization allowed the same number of stored
        // entries: rank r_glr with r_glr * 2 nv = storage.
        const int rank_glr = std::max(1, static_cast<int>(storage / ( 2LL * nv )));
        const LowRank G = truncated_svd(A, 0.0, rank_glr);
        const double err_glr = ( A - G.to_dense() ).norm() / norm_A;

        // The same GLR built from the BRLR applies alone (no kernel access).
        const LowRank G_rsvd = randomized_svd(r.matrix, rank_glr);
        const double err_rsvd = ( A - G_rsvd.to_dense() ).norm() / norm_A;

        std::vector<int> ranks;
        for ( const BlockLowRank::Block& blk : r.matrix.blocks() )
        {
            ranks.push_back(blk.rank());
        }
        std::printf("  %.0e    %4.1f%%     %.2e       %4d   %.2e           %.2e   %s\n",
                    rtol, 100.0 * storage / ( double(nv) * nv ), err,
                    rank_glr, err_glr, err_rsvd,
                    r.all_converged ? "" : "(!) not converged");

        if ( rtol == headline_rtol )
        {
            B_headline = r.matrix;
            ranks_headline = ranks;
        }
    }
    std::printf("\nheadline rtol %.0e: median block rank %d of up to %d sources per block\n",
                headline_rtol, median_int(ranks_headline), cutoff);

    // ---- Figure: the partition, one block's target set and ellipsoids ----
    {
        // The highlighted block: the one containing the vertex nearest (0.62, 0.33).
        int highlight = 0;
        {
            int nearest = 0;
            ( vertices.colwise() - Eigen::Vector2d(0.62, 0.33) )
                .colwise().squaredNorm().minCoeff(&nearest);
            for ( size_t pp = 0; pp < partition.size(); ++pp )
            {
                for ( int jj : partition[pp] )
                {
                    if ( jj == nearest )
                    {
                        highlight = static_cast<int>(pp);
                    }
                }
            }
        }
        const BlockLowRank::Block& blk = B_headline.blocks()[highlight];

        Plot2D fig;
        fig.add(Box{Eigen::Vector2d(0, 0), Eigen::Vector2d(1, 1)},
                Style{with_alpha(colors::black(), 0.4), 1.0, colors::transparent()});
        // The highlighted block's target set, as soft halos behind the
        // source dots (sources and targets are the same vertices here).
        for ( int tt : blk.target_ids )
        {
            fig.add_marker(vertices.col(tt), 3.6,
                           Style{colors::transparent(), 0.0,
                                 with_alpha(colors::sky_blue(), 0.55)});
        }
        // All sources, colored by subdomain.
        for ( size_t pp = 0; pp < partition.size(); ++pp )
        {
            const Color c = palette_color(static_cast<int>(pp));
            for ( int jj : partition[pp] )
            {
                fig.add_marker(vertices.col(jj), 1.6, Style{c, 0.8, c});
            }
        }
        // The highlighted block's sources and a sample of their support
        // ellipsoids (every 12th, to keep the union readable).
        for ( size_t ss = 0; ss < blk.source_ids.size(); ++ss )
        {
            fig.add_marker(vertices.col(blk.source_ids[ss]), 2.2,
                           Style{colors::black(), 1.0, colors::black()});
            if ( ss % 24 == 0 )
            {
                for ( const Ellipsoid& e : K.target_support(vertices.col(blk.source_ids[ss])) )
                {
                    fig.add(e, 1.0, Style{with_alpha(colors::black(), 0.5), 1.0,
                                          colors::transparent()});
                }
            }
        }
        fig.save_png("01_partition_and_target_set.png", 560);
        std::printf("highlighted block: %zu sources couple to %zu of %d targets\n",
                    blk.source_ids.size(), blk.target_ids.size(), nv);

        char caption[700];
        std::snprintf(caption, sizeof(caption),
            "**The source partition and one block's reach.** Eight subdomains "
            "(colors) from recursive median bisection of the source points. For "
            "the highlighted block (black): the thin ellipses are a sample of its "
            "support ellipsoids (every 24th source), and the blue halos mark the "
            "%zu of %d target vertices where its kernel columns can be nonzero. "
            "Every entry outside that set is exactly zero — the block sparsity is "
            "lossless, and all approximation error comes from the per-block "
            "truncation.",
            blk.target_ids.size(), nv);
        write_caption("01_partition_and_target_set", caption);
    }

    // ---- Figure: per-block rank as a map over the source domain ----
    {
        Eigen::VectorXd rank_field(nv);
        for ( size_t pp = 0; pp < partition.size(); ++pp )
        {
            for ( int jj : partition[pp] )
            {
                rank_field(jj) = ranks_headline[pp];
            }
        }
        Plot2D fig;
        FieldOptions opts;
        opts.vmin = 0.0;
        opts.vmax = rank_field.maxCoeff();
        draw_cg1_field(fig, F->target_mesh(), rank_field, opts);
        add_colorbar(fig, opts.vmin, opts.vmax, "%.0f");
        fig.save_png("02_block_ranks.png", 560);

        write_caption("02_block_ranks",
            "**Per-block compressed rank at the headline tolerance (rtol 1e-2), "
            "as a map over the source domain.** Each subdomain is colored by the "
            "rank of its block after Frobenius truncation. Rank tracks how much "
            "the kernel varies across the block relative to the ellipsoid size: "
            "the middle columns, where the support ellipsoids are large and "
            "rotate quickly, need noticeably higher rank than the calmer left "
            "and right edges.");
    }

    // ---- Figures: column-error maps, BRLR vs GLR at equal storage ----
    {
        const int rank_glr =
            std::max(1, static_cast<int>(B_headline.storage_entries() / ( 2LL * nv )));
        const Eigen::MatrixXd B_dense = B_headline.to_dense();
        const Eigen::MatrixXd G_dense = truncated_svd(A, 0.0, rank_glr).to_dense();
        Eigen::VectorXd err_brlr(nv), err_glr(nv);
        for ( int jj = 0; jj < nv; ++jj )
        {
            const double d = std::max(col_norms(jj), norm_floor);
            err_brlr(jj) = std::min(( B_dense.col(jj) - A.col(jj) ).norm() / d, 1.0);
            err_glr(jj)  = std::min(( G_dense.col(jj) - A.col(jj) ).norm() / d, 1.0);
        }
        FieldOptions opts;
        opts.vmin = 0.0;
        opts.vmax = 1.0;
        Plot2D fig_b;
        draw_cg1_field(fig_b, F->target_mesh(), err_brlr, opts);
        add_colorbar(fig_b, 0.0, 1.0, "%.1f");
        fig_b.save_png("03_col_err_brlr.png", 560);
        Plot2D fig_g;
        draw_cg1_field(fig_g, F->target_mesh(), err_glr, opts);
        add_colorbar(fig_g, 0.0, 1.0, "%.1f");
        fig_g.save_png("04_col_err_glr_equal_storage.png", 560);

        char caption[700];
        std::snprintf(caption, sizeof(caption),
            "**BRLR relative column error at rtol 1e-2.** The error of each "
            "kernel column, plotted at its source location (relative per column, "
            "denominators floored at 1%% of the largest column norm, clipped at "
            "1; shared 0..1 color scale with the next figure). The per-block "
            "relative tolerance spreads accuracy uniformly — including the "
            "small-amplitude columns near the boundary.");
        write_caption("03_col_err_brlr", caption);

        std::snprintf(caption, sizeof(caption),
            "**Global low rank with the same total storage (rank %d): the "
            "locality comparison.** A single Frobenius-optimal factorization "
            "spends its budget on the dominant interior structure and "
            "under-serves the small-amplitude, fast-rotating columns near the "
            "boundary — the bright ring. The block format protects them by "
            "construction: each block meets the tolerance relative to its own "
            "norm.",
            rank_glr);
        write_caption("04_col_err_glr_equal_storage", caption);
    }

    return 0;
}

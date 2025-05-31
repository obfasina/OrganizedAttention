using Markdown
using InteractiveUtils
using MultiscaleGraphSignalTransforms, Plots, LinearAlgebra
using NPZ
using PyCall
using Plots
using Random
using Distances
using ProgressMeter
using StatsBase
np=pyimport("numpy")
pysys = pyimport("sys")
push!(pysys["path"], @__DIR__)

function get_Walsh_partition(A)
    matrix=A
    m,n = size(matrix);
    Gm = gpath(m);
    W = Gm.W;
    L = diagm(sum(W; dims = 1)[:]) - W;
    𝛌, 𝚽 = eigen(L);
    𝚽 = 𝚽 .* sign.(𝚽[1,:])';
    GProws = partition_tree_fiedler(Gm; swapRegion = false);
    # column tree
    Gn = gpath(n);
    W = Gn.W;
    L = diagm(sum(W; dims = 1)[:]) - W;
    𝛌, 𝚽 = eigen(L);
    𝚽 = 𝚽 .* sign.(𝚽[1,:])';
    GPcols = partition_tree_fiedler(Gn; swapRegion = false);
    return GProws,GPcols
end

function get_Walsh_coefficients(matrix,GProws,GPcols)
    m,n=size(matrix)
    dvec, BSrows, BScols = ghwt_bestbasis_2d(matrix,GProws,GPcols);
    indices = sortperm(reshape(abs.(dvec[:]),m*n,1), rev=true,dims = 1);
    return dvec, indices, BSrows, BScols
end

function get_BS_coefficients_1D(data,GP,BS)
    n=length(data)
  G = gpath(n)
  G.f = data
  dmatrix = ghwt_analysis!(G, GP = GP)
    dvec = dmatrix2dvec(dmatrix, GP, BS)
  return dvec
end

function get_ghwt_synthesis(dvec,GP,BS)
      	f = ghwt_synthesis(dvec, GP, BS)
	return f
end

 

function get_scores_thresholded_coeffs!(score,finalindex,listinds,dvec,indices,compare_matrix,GP_cols,GP_rows,BS_cols,BS_rows)
    dvec_rec=zeros(size(dvec))
    this_index=indices[1:finalindex]
    dvec_rec[this_index]=dvec[this_index]
    rec=ghwt_synthesis_2d(dvec_rec,GP_rows,GP_cols,BS_rows,BS_cols)
    score[1]=opnorm(rec)
    score[2]=opnorm(compare_matrix-rec[listinds,listinds])
end

function from_matrix_get_score(finalindex,listinds,datamatrix,GProws,GPcols,normalization)
    score=zeros(2)
    dvec,indices,BScols,BSrows=get_Walsh_coefficients(datamatrix,GProws,GPcols)
    matrix_main=datamatrix[listinds,listinds]
    get_scores_thresholded_coeffs!(score,finalindex,listinds,dvec,indices,matrix_main,GPcols,GProws,BScols,BSrows)
    return score./normalization
end

function get_inverse_permutation(perm)
    inverse=np.argsort(np.asarray(perm)) .+ 1
    return inverse
end
function build_kernel_permutations(A,oldtreebuilding::Bool = false)
    #heatmap(data_permuted)
    if oldtreebuilding
        qrun_permuted=quaest.main(A,oldtreebuilding)
    else
        qrun_permuted=quaest.main(A)
    end

    row_tree=qrun_permuted.row_trees[end]
    col_tree=qrun_permuted.col_trees[end]

    coifman_col_order=[x.elements[1] for x in col_tree.dfs_leaves()] .+1
    coifman_row_order=[x.elements[1] for x in row_tree.dfs_leaves()] .+1

    data_afterquest=A[coifman_row_order,coifman_col_order]
    return data_afterquest,coifman_col_order,coifman_row_order
end

function plot_single(xdata,threeDtensor,arr_w,switch)
    nplots=size(threeDtensor)[2]

    # Generate 5 colors from the "hot" palette
    c = palette(:hot, nplots+3)

    # Initialize a plot
    p = plot()
    if switch==1
        titlelabel="relative norm of reconstruction"
    elseif switch ==2
        titlelabel="relative norm of residual"
    end

    # Plot each column (each series) with a different color
    for i in 1:nplots
        plot!(p, xdata, threeDtensor[:, i,switch], 
        color = c[i+1], 
        label = "frequency w = $(arr_w[i])",
        xlabel= "fraction of coefficients kept",
        title= titlelabel
        )
    end
    display(p)
end


function plot_sidebyside(xdata,threeDtensor,arr_w)
    nplots=size(threeDtensor)[2]
    c = palette(:hot, nplots+3)

    # Build a 1×2 layout, i.e. two side-by-side subplots
    p = plot(layout=(1,2), size=(800,450))

    for i in 1:nplots
        # Plot "first output" (A[:, i, 1]) in left subplot
        plot!(p[1],
          xdata,
          threeDtensor[:, i, 1],
          color = c[i+1],
          label = "frequency w=$(arr_w[i])",
          xlabel="fraction of coefficients kept",
          title= "relative norm of reconstruction"
        )

        # Plot "second output" (A[:, i, 2]) in right subplot
        plot!(p[2],
          xdata,
          threeDtensor[:, i, 2],
          color = c[i+1],
          label = "frequency w=$(arr_w[i])",
          xlabel="fraction of coefficients kept",
          title= "relative norm of the residual"
        )
    end
    display(p)
end


function plot_singlefrequency(freq_index,switch, logscale::Bool = false)
    if switch==1
        titlelabel="Frequency w=$(powers[freq_index])"
    elseif switch ==2
        titlelabel="Frequency w=$(powers[freq_index])"
    end
    yscale_choice = logscale ? :log10 : :identity
    plot(pertoplot,score_original[:,freq_index,switch],yscale=yscale_choice,label="original",xlabel="fraction of coefficients kept")
    plot!(pertoplot,score_permuted[:,freq_index,switch],yscale=yscale_choice,label="after applying permutation")
    plot!(pertoplot,score_afterquest[:,freq_index,switch],yscale=yscale_choice,label="after applying the questionnaire")
    plot!(title=titlelabel)
end


function single_freq_heatmaps(index_freq,switch)

    data_osc=np.imag(exp.(1im*2*pi * powers[index_freq] .* Dm))
    data_decay= np.nan_to_num(data_osc ./ Dm)
    npoints=size(data_decay)[1]
    perm=sample(range(1,npoints),npoints,replace=false)
    ### Change this line to go switch between pure oscillation and oscillation+decay
    if switch>0
        matrix_main=data_osc
        C=1
    else
        matrix_main=data_decay
        asas=vec(np.nan_to_num(log.(abs.(data_decay)),posinf=0,neginf=0))
        C=exp(mean(asas)+3*std(asas))
        println("color cutoff= ",C)
        #C=maximum(abs.(data_decay))
    end
    data_permuted=matrix_main[perm,perm]
    data_afterquest,coifman_col_order,coifman_row_order=build_kernel_permutations(matrix_main)
    clrange = (-C, C)
    pcolor = heatmap( fill(NaN,1,1);
                  clims=clrange,  # same color limits
                  colorbar=true,
                  framestyle=:none,  # hide axes
                  )

    # Custom layout: 
    #  - The first part is a grid(1,3) for the 3 heatmaps
    #  - {0.8w} means "occupy 80% of the total width"
    #  - Then a second cell for the colorbar subplot, with {0.2w} = 20% width
    layout_ = @layout([ grid(1,3){0.95w}  c{0.05w} ])
    p = plot(
        heatmap(matrix_main, colorbar=false,title="Original data",clims=clrange),
        heatmap(data_permuted, colorbar=false,title="After scrambling",clims=clrange),
        heatmap(data_afterquest, colorbar=false,title="After questionnaire",clims=clrange),
        pcolor,
        layout = layout_,
        size = (1200, 300)    # optional size
    )
end

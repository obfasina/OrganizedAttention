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
using JLD2
np=pyimport("numpy")
pysys = pyimport("sys")
push!(pysys["path"], @__DIR__)
quaest=pyimport("Apply_questionnaire_to_data")
Random.seed!(1234)


### Let's make an ellipse that will be both the 'sources' and 'receivers' of our problem

Npoints=500
randomsel=sort(rand((Npoints)))
a=10
b=5
X = 0.05 .* a .* cos.(2*pi*randomsel)
Y = 0.05 .* b .* sin.(2*pi*randomsel)
Z=vcat(X',Y')
Dm=pairwise(Euclidean(),Z,dims=2)
data_osc=np.imag(exp.(1im*2*pi *5 .* Dm))
data_decay= np.nan_to_num(data_osc ./ Dm)


### Change this line to go switch between pure oscillation and oscillation+decay
matrix_main=data_osc
#matrix_main=data_decay
heatmap(reverse(matrix_main,dims=1))

### Now, let's introduce a permutation to the rows and columns

perm=sample(range(1,Npoints),Npoints,replace=false)
data_permuted=matrix_main[perm,perm]
function get_inverse_permutation(perm)
    inverse=np.argsort(np.asarray(perm)) .+ 1
    return inverse
end
invperm=get_inverse_permutation(perm)
heatmap(reverse(data_permuted,dims=1))

### Now, let's apply raphy's questionnaire method twice, once for the permuted, once for the unpermuted kernel

function heatmaps_sidebyside(threeDtensor)
    nplots=size(threeDtensor)[end]
    c = palette(:hot, nplots+3)

    # Build a 1×2 layout, i.e. two side-by-side subplots
    p = plot(layout=(1,nplots), size=(900,400))

    for i in 1:nplots
        # Plot "first output" (A[:, i, 1]) in left subplot
        heatmap!(p[i],
          reverse(threeDtensor[:, :, i],dims=1),
        )
        # Plot "second output" (A[:, i, 2]) in right subplot
    end
    display(p)
end

heatmaps_sidebyside(cat(matrix_main,data_permuted,dims=3))

function build_kernel_permutations(A,oldtreebuilding::Bool = false)
    if oldtreebuilding
        qrun_permuted=quaest.main(A,oldtreebuilding)
    else
        qrun_permuted=quaest.main(A)
    end

    row_tree=qrun_permuted.row_trees[end]
    col_tree=qrun_permuted.col_trees[end]

    coifman_col_order=[x.elements[1] for x in col_tree.dfs_leaves()] .+ 1
    coifman_row_order=[x.elements[1] for x in row_tree.dfs_leaves()] .+ 1

    data_afterquest=A[coifman_row_order,coifman_col_order]
    return data_afterquest,coifman_col_order,coifman_row_order, row_tree
end

data_original_afterquest,raphy_col_order1,raphy_row_order1,tree_original=build_kernel_permutations(matrix_main)
data_permuted_afterquest,raphy_col_order2,raphy_row_order2, tree_permuted=build_kernel_permutations(data_permuted)
data_original_afterquest_old,raphy_col_order1_old,raphy_row_order1_old,tree_original_old=build_kernel_permutations(matrix_main,true)
data_permuted_afterquest_old,raphy_col_order2_old,raphy_row_order2_old, tree_permuted_old=build_kernel_permutations(data_permuted,true)
data_permuted_afterquest=reverse(data_permuted_afterquest)
data_permuted_afterquest_old=reverse(data_permuted_afterquest_old)
raphy_col_order2=reverse(raphy_col_order2)
raphy_col_order2_old=reverse(raphy_col_order2_old)

Old_matrices=cat(data_original_afterquest_old,data_permuted_afterquest_old,dims=3)
New_matrices=cat(data_original_afterquest,data_permuted_afterquest,dims=3)

inv_quest_original_afterquest=get_inverse_permutation(raphy_col_order1)
inv_quest_permuted_afterquest=get_inverse_permutation(raphy_col_order2)[invperm]
inv_quest_original_afterquest_old=get_inverse_permutation(raphy_col_order1_old)
inv_quest_permuted_afterquest_old=get_inverse_permutation(raphy_col_order2_old)[invperm]
vectest=round.(Int,LinRange(1,Npoints,Npoints))

heatmaps_sidebyside(Old_matrices)
heatmaps_sidebyside(New_matrices)

heatmap(data_original_afterquest)
heatmap(data_permuted_afterquest)
quaest.plot_tree(tree_original)
quaest.plot_tree(tree_permuted)

#@save "./basic_workspace_variables_postquestionnaire.jld" Z Dm matrix_main perm randomsel data_original_afterquest data_original_afterquest_old data_permuted data_permuted_afterquest data_permuted_afterquest_old
### now lets see how sparse or non-sparse these different matrices are in terms of the
### Walsh best basis decomposition

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

GProws_original,GPcols_original=get_Walsh_partition(matrix_main)
GProws_permuted,GPcols_permuted=get_Walsh_partition(data_permuted)
GProws_afterquest_o,GPcols_afterquest_o=get_Walsh_partition(data_original_afterquest)
GProws_afterquest_p,GPcols_afterquest_p=get_Walsh_partition(data_permuted_afterquest)
GProws_afterquest_o_old,GPcols_afterquest_o_old=get_Walsh_partition(data_original_afterquest_old)
GProws_afterquest_p_old,GPcols_afterquest_p_old=get_Walsh_partition(data_permuted_afterquest_old)
m,n=size(matrix_main)

function get_Walsh_coefficients(matrix,GProws,GPcols)
    dvec, BSrows, BScols = ghwt_bestbasis_2d(matrix,GProws,GPcols);
    indices = sortperm(reshape(abs.(dvec[:]),m*n,1), rev=true,dims = 1);
    return dvec, indices, BSrows, BScols
end

dvec_original,indices_original,BSrows_original,BScols_original=get_Walsh_coefficients(matrix_main,GProws_original,GPcols_original)
dvec_permuted,indices_permuted,BSrows_permuted,BScols_permuted=get_Walsh_coefficients(data_permuted,GProws_permuted,GPcols_permuted)

dvec_afterquest_o_old,indices_afterquest_o_old,BSrows_afterquest_o_old,BScols_afterquest_o_old=get_Walsh_coefficients(data_original_afterquest_old,GProws_afterquest_o_old,GPcols_afterquest_o_old)
dvec_afterquest_p_old,indices_afterquest_p_old,BSrows_afterquest_p_old,BScols_afterquest_p_old=get_Walsh_coefficients(data_permuted_afterquest_old,GProws_afterquest_p_old,GPcols_afterquest_p_old)
dvec_afterquest_o,indices_afterquest_o,BSrows_afterquest_o,BScols_afterquest_o=get_Walsh_coefficients(data_original_afterquest,GProws_afterquest_o,GPcols_afterquest_o)
dvec_afterquest_p,indices_afterquest_p,BSrows_afterquest_p,BScols_afterquest_p=get_Walsh_coefficients(data_permuted_afterquest,GProws_afterquest_p,GPcols_afterquest_p)
dvec_o_mixed_perm,indices_o_mixed_perm,BSrows_mixed_perm,BScols_mixed_perm=get_Walsh_coefficients(matrix_main,GProws_permuted,GPcols_permuted)
dvec_old_mixed,indices_old_mixed,BSrows_old_mixed,BScols_old_mixed=get_Walsh_coefficients(data_original_afterquest_old,GProws_afterquest_p_old,GPcols_afterquest_p_old)
dvec_mixed_o_afterquest,indices_mixed_o_afterquest,BSrows_mixed_o_afterquest,BScols_mixed_o_afterquest=get_Walsh_coefficients(matrix_main,GProws_afterquest_o,GPcols_afterquest_o)


#pertoplot=np.hstack(((np.linspace(0,1-np.exp(np.linspace(-8,0,18))[end-1],20),(1 .- reverse(np.exp(np.linspace(-8,0,18))))[3:end])))
nprobes=40
pertoplot=1 ./(1 .+ (cot.( pi.*LinRange(0.01,0.49,nprobes))).^ 3)
vecnum=round.(Int,Npoints*Npoints .*pertoplot)
datanorm=opnorm(matrix_main,2)
### Let's plot how well does the hierarchical decomposition of one matrix
### works on another matrix

function get_scores_thresholded_coeffs!(score,finalindex,listinds,dvec,indices,GP_cols,GP_rows,BS_cols,BS_rows)
    dvec_rec=zeros(size(dvec))
    this_index=indices[1:finalindex]
    dvec_rec[this_index]=dvec[this_index]
    rec=ghwt_synthesis_2d(dvec_rec,GP_rows,GP_cols,BS_rows,BS_cols)[listinds,listinds]
    score[1]=opnorm(rec)
    score[2]=opnorm(matrix_main-rec)
    return rec
end

function from_matrix_get_score(finalindex,listinds,datamatrix,GProws,GPcols)
    score=zeros(2)
    dvec,indices,BScols,BSrows=get_Walsh_coefficients(datamatrix,GProws,GPcols)
    Rec=get_scores_thresholded_coeffs!(score,finalindex,listinds,dvec,indices,GPcols,GProws,BScols,BSrows)
    return score./datanorm, Rec
end

function score_matrix_mismatch(finalindex,listinds,datamatrix,GProws,GPcols,BSrows,BScols)
    score=zeros(2)
    dvec,indices,ucols,urows=get_Walsh_coefficients(datamatrix,GProws,GPcols)
    Rec=get_scores_thresholded_coeffs!(score,finalindex,listinds,dvec,indices,GPcols,GProws,BScols,BSrows)
    return score./datanorm, Rec
end

function get_explicit_Walsh_basis(datamatrix,GProws,GPcols)
    dvec,indices,BScols,BSrows=get_Walsh_coefficients(datamatrix,GProws,GPcols)
    m,n=size(dvec)
    Colbasis=zeros(size(dvec))
    Rowbasis=zeros(size(dvec))
    IdentCol=np.eye(n)
    IdentRow=np.eye(m)
    dvec_row=zeros(m,1)
    dvec_col=zeros(n,1)
    colindex=div.((indices .- 1),m) .+ 1
    rowindex=(indices_original .- 1) .% m .+ 1
    for j in 1:size(dvec)[2]
        dvec_row[:,1]=IdentRow[:,colindex[j]]
        Colbasis[:,j]=ghwt_synthesis(dvec_row,GPcols,BScols)
    end
    for j in 1:size(dvec)[1]
        dvec_col[:,1]=IdentCol[:,rowindex[j]]
        Rowbasis[:,j]=ghwt_synthesis(dvec_col,GProws,BSrows)
    end
    return Rowbasis,Colbasis
end

### Initialize the score arrays

score_afterquest_o_old=zeros(size(vecnum)[1],2)
score_afterquest_p_old=zeros(size(vecnum)[1],2)
score_o_mixed_perm=zeros(size(vecnum)[1],2)
score_old_mixed=zeros(size(vecnum)[1],2)
score_o_mixed_afterquest=zeros(size(vecnum)[1],2)
score_original=zeros(size(vecnum)[1],2)
score_permuted=zeros(size(vecnum)[1],2)
score_afterquest_o=zeros(size(vecnum)[1],2)
score_afterquest_p=zeros(size(vecnum)[1],2)
score_mixed_afterquest_perm=zeros(size(vecnum)[1],2)

### Update them in place
for k in eachindex(vecnum)
    score_afterquest_o_old[k,:],rec_afterquest_o_old=from_matrix_get_score(vecnum[k],inv_quest_original_afterquest_old,data_original_afterquest_old,GProws_afterquest_o_old,GPcols_afterquest_o_old)
    score_afterquest_p_old[k, :],rec_afterquest_p_old=from_matrix_get_score(vecnum[k],inv_quest_permuted_afterquest_old,data_permuted_afterquest_old,GProws_afterquest_p_old,GPcols_afterquest_p_old)
    score_afterquest_o[k,:],rec_afterquest_o=from_matrix_get_score(vecnum[k],inv_quest_original_afterquest,data_original_afterquest,GProws_afterquest_o,GPcols_afterquest_o)
    score_afterquest_p[k,:],rec_afterquest_p=from_matrix_get_score(vecnum[k],inv_quest_permuted_afterquest,data_permuted_afterquest,GProws_afterquest_p,GPcols_afterquest_p)
    score_old_mixed[k,:],rec_old_mixed=from_matrix_get_score(vecnum[k],inv_quest_original_afterquest_old,data_permuted_afterquest_old,GProws_afterquest_p_old,GPcols_afterquest_p_old)
    score_o_mixed_perm[k,:],rec_o_mixed_perm=from_matrix_get_score(vecnum[k],vectest,matrix_main,GProws_permuted,GPcols_permuted)
    score_o_mixed_afterquest[k,:],rec_o_mixed_afterquest=from_matrix_get_score(vecnum[k],vectest,matrix_main,GProws_afterquest_o,GPcols_afterquest_o)
    score_original[k,:],rec_original=from_matrix_get_score(vecnum[k],vectest,matrix_main,GProws_original,GPcols_original)
    score_permuted[k,:],rec_perm=from_matrix_get_score(vecnum[k],invperm,data_permuted,GProws_permuted,GPcols_permuted)
    score_mixed_afterquest_perm[k,:],rec_mixed_afterquest_perm=from_matrix_get_score(vecnum[k],inv_quest_original_afterquest,data_permuted_afterquest,GProws_afterquest_p,GPcols_afterquest_p)
end

function plot_sidebyside(xdata,threeDtensor,switchplot=false)
    val=2-Int.(switchplot)
    nplots=size(threeDtensor)[3]
    c = palette(:hot, nplots+3)
    # Build a 1×2 layout, i.e. two side-by-side subplots
    p = plot(layout=(1,1), size=(450,450))

    for i in 1:nplots
        plot!(p[1],
          xdata,
          threeDtensor[:, val, i],
          xlabel="fraction of coefficients kept",
          ylabel= "relative norm of the residual"
        )
    end
    display(p)
end

plot_sidebyside(pertoplot,cat(score_original,score_permuted,dims=3))
plot_sidebyside(pertoplot,cat(score_afterquest_o_old,score_afterquest_p_old,dims=3))
plot_sidebyside(pertoplot,cat(score_afterquest_o,score_afterquest_p,dims=3))
plot_sidebyside(pertoplot,cat(score_original,score_o_mixed_perm,dims=3))
plot_sidebyside(pertoplot,cat(score_original,score_old_mixed,dims=3))
plot_sidebyside(pertoplot,cat(score_original,score_o_mixed_afterquest,dims=3))
plot_sidebyside(pertoplot,cat(score_afterquest_o,score_afterquest_p,score_mixed_afterquest_perm,dims=3),true)


kk=8

_,rec_afterquest_o_old=from_matrix_get_score(vecnum[kk],inv_quest_original_afterquest_old,data_original_afterquest_old,GProws_afterquest_o_old,GPcols_afterquest_o_old)
_,rec_afterquest_p_old=from_matrix_get_score(vecnum[kk],inv_quest_permuted_afterquest_old,data_permuted_afterquest_old,GProws_afterquest_p_old,GPcols_afterquest_p_old)
_,rec_afterquest_o=from_matrix_get_score(vecnum[kk],inv_quest_original_afterquest,data_original_afterquest,GProws_afterquest_o,GPcols_afterquest_o)
_,rec_afterquest_p=from_matrix_get_score(vecnum[kk],inv_quest_permuted_afterquest,data_permuted_afterquest,GProws_afterquest_p,GPcols_afterquest_p)
_,rec_old_mixed=from_matrix_get_score(vecnum[kk],inv_quest_original_afterquest_old,data_permuted_afterquest_old,GProws_afterquest_p_old,GPcols_afterquest_p_old)
_,rec_o_mixed_perm=from_matrix_get_score(vecnum[kk],vectest,matrix_main,GProws_permuted,GPcols_permuted)
_,rec_o_mixed_afterquest=from_matrix_get_score(vecnum[kk],vectest,matrix_main,GProws_afterquest_o,GPcols_afterquest_o)
_,rec_original=from_matrix_get_score(vecnum[kk],vectest,matrix_main,GProws_original,GPcols_original)
_,rec_perm=from_matrix_get_score(vecnum[kk],invperm,data_permuted,GProws_permuted,GPcols_permuted)
_,rec_mixed_afterquest_perm=from_matrix_get_score(vecnum[kk],inv_quest_original_afterquest,data_permuted_afterquest,GProws_afterquest_p,GPcols_afterquest_p)



heatmap(rec_afterquest_p_old-matrix_main)
heatmap(rec_afterquest_p-matrix_main)

#### now,let's consider a different type of basis mismatch


score_o_mixed_perm2=zeros(size(vecnum)[1],2)
score_old_mixed2=zeros(size(vecnum)[1],2)
score_o_mixed_afterquest2=zeros(size(vecnum)[1],2)
score_mixed_afterquest_perm2=zeros(size(vecnum)[1],2)

### Update them in place
for k in eachindex(vecnum)
    score_old_mixed2[k,:],rec_old_mixed=score_matrix_mismatch(vecnum[k],inv_quest_original_afterquest_old,data_permuted_afterquest_old,GProws_afterquest_p_old,GPcols_afterquest_p_old,BSrows_afterquest_o_old,BScols_afterquest_o_old)
    score_o_mixed_perm2[k,:],rec_o_mixed_perm=score_matrix_mismatch(vecnum[k],vectest,matrix_main,GProws_permuted,GPcols_permuted,BSrows_permuted,BScols_permuted)
    score_o_mixed_afterquest2[k,:],rec_o_mixed_afterquest=score_matrix_mismatch(vecnum[k],vectest,matrix_main,GProws_afterquest_o,GPcols_afterquest_o,BSrows_afterquest_o,BScols_afterquest_o)
    score_mixed_afterquest_perm2[k,:],rec_mixed_afterquest_perm=score_matrix_mismatch(vecnum[k],inv_quest_original_afterquest,data_permuted_afterquest,GProws_afterquest_p,GPcols_afterquest_p,BSrows_afterquest_o,BScols_afterquest_o)
end

plot_sidebyside(pertoplot,cat(score_original,score_o_mixed_perm2,dims=3))
plot_sidebyside(pertoplot,cat(score_original,score_old_mixed2,dims=3))
plot_sidebyside(pertoplot,cat(score_original,score_o_mixed_afterquest2,dims=3))
plot_sidebyside(pertoplot,cat(score_afterquest_o,score_afterquest_p,score_mixed_afterquest_perm2,dims=3),true)

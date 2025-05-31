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
#matrix_main=data_osc
matrix_main=data_decay
heatmap(matrix_main)

### Now, let's introduce a permutation to the rows and columns

perm=sample(range(1,Npoints),Npoints,replace=false)
function get_inverse_permutation(perm)
    inverse=np.argsort(np.asarray(perm)) .+ 1
    return inverse
end
invperm=get_inverse_permutation(perm)
data_permuted=matrix_main[perm,perm]
heatmap(data_permuted)

qrun_permuted=quaest.main(data_permuted)

row_tree=qrun_permuted.row_trees[end]
col_tree=qrun_permuted.col_trees[end]

coifman_col_order=[x.elements[1] for x in col_tree.dfs_leaves()] .+ 1
coifman_row_order=[x.elements[1] for x in row_tree.dfs_leaves()] .+ 1
inv_quest_order=get_inverse_permutation(coifman_col_order)
composite_perm=inv_quest_order[invperm]
data_afterquest=data_permuted[coifman_row_order,coifman_col_order]
heatmap(data_afterquest)

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
GProws_afterquest,GPcols_afterquest=get_Walsh_partition(data_afterquest)

function get_Walsh_coefficients(matrix,GProws,GPcols)
    dvec, BSrows, BScols = ghwt_bestbasis_2d(matrix,GProws,GPcols);
    indices = sortperm(reshape(abs.(dvec[:]),m*n,1), rev=true,dims = 1);
    return dvec, indices, BSrows, BScols
end

function get_scores_thresholded_coeffs!(score,finalindex,listinds,dvec,indices,GP_cols,GP_rows,BS_cols,BS_rows)
    dvec_rec=zeros(size(dvec))
    this_index=indices[1:finalindex]
    dvec_rec[this_index]=dvec[this_index]
    rec=ghwt_synthesis_2d(dvec_rec,GP_rows,GP_cols,BS_rows,BS_cols)
    score[1]=opnorm(rec)
    score[2]=opnorm(matrix_main-rec[listinds,listinds])
end

function from_matrix_get_score(finalindex,listinds,datamatrix,GProws,GPcols)
    score=zeros(2)
    dvec,indices,BScols,BSrows=get_Walsh_coefficients(datamatrix,GProws,GPcols)
    get_scores_thresholded_coeffs!(score,finalindex,listinds,dvec,indices,GPcols,GProws,BScols,BSrows)
    return score./datanorm
end

vectest=round.(Int,LinRange(1,Npoints,Npoints))
dvec_original,indices_original,BSrows_original,BScols_original=get_Walsh_coefficients(matrix_main,GProws_original,GPcols_original)
dvec_permuted,indices_permuted,BSrows_permuted,BScols_permuted=get_Walsh_coefficients(data_permuted,GProws_permuted,GPcols_permuted)
dvec_afterquest,indices_afterquest,BSrows_afterquest,BScols_afterquest=get_Walsh_coefficients(data_afterquest,GProws_afterquest,GPcols_afterquest)

heatmap(dvec_original)
heatmap(dvec_permuted)
heatmap(dvec_afterquest)

#pertoplot=np.hstack(((np.linspace(0,1-np.exp(np.linspace(-8,0,18))[end-1],20),(1 .- reverse(np.exp(np.linspace(-8,0,18))))[3:end])))
pertoplot=1 ./(1 .+ (cot.( pi.*LinRange(0.01,0.49,40))).^ 3)
vecnum=round.(Int,Npoints*Npoints .*pertoplot)
datanorm=opnorm(matrix_main,2)
score_original=zeros(size(vecnum)[1],2)
score_permuted=zeros(size(vecnum)[1],2)
score_afterquest=zeros(size(vecnum)[1],2)


for k in eachindex(vecnum)
    score_original[k,:]=from_matrix_get_score(vecnum[k],vectest,matrix_main,GProws_original,GPcols_original)
    score_permuted[k,:]=from_matrix_get_score(vecnum[k],invperm,data_permuted,GProws_permuted,GPcols_permuted)
    score_afterquest[k,:]=from_matrix_get_score(vecnum[k],composite_perm,data_afterquest,GProws_afterquest,GPcols_afterquest)
end

plot(pertoplot,score_original[1:end,1],label="original",xlabel="fraction of coefficients kept")
plot!(pertoplot,score_permuted[1:end,1],label="after applying permutation")
plot!(pertoplot,score_afterquest[1:end,1],label="after applying the questionnaire")
plot!(title="Relative norm of the reconstruction")

plot(pertoplot,score_original[1:end,2],label="original",xlabel="fraction of coefficients kept")
plot!(pertoplot,score_permuted[1:end,2],label="after applying permutation")
plot!(pertoplot,score_afterquest[1:end,2],label="after applying the questionnaire")
plot!(title="Relative norm of data minus reconstruction")

#Find top coefficients
#indices = sortperm(reshape(abs.(dvec[:]),m*n,1), rev=true,dims = 1);
%%%%load M_cat in the output folder
%%%%change nb_samples and nb_features
nb_samples=6;
nb_features=8;

[ V ,  qDs ]= joint_diag(M_cat,1.0e-8);

inds_max=[];
all_diags=[];
errors=[];
for i=1:nb_samples
    istart=nb_features*(i-1)+1;
    iend=nb_features*i;
    approx_diag=qDs(:,istart:iend);
    this_diag=diag(approx_diag);
    D=diag(this_diag);
    approx_matrix=V*D*inv(V);
    this_diag=abs(this_diag);
    all_diags=[all_diags, this_diag];
end

figure(1), imagesc(abs(V))
figure(2),imagesc(all_diags)

scriptPath = mfilename('fullpath');
path_output = fileparts(scriptPath);
save(strcat(path_output,'/output/V.mat'))
save(strcat(path_output,'/output/all_diags.mat'))
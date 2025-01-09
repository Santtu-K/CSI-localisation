clear
close all;

cov = 1;

fileID_z = fopen("res_r_real2.text", "a");
fprintf(fileID_z,'\nTime: %s\n', datestr(datetime("now")));

layer_configs = [
    [512,256,128,64,32,16]; % [256,128,32,16,8,4]; 
]';

grid = 3;

if cov == 1
    loadstr = strcat("..\randomUE_d_grid", int2str(grid), "_real_cov.mat");
    loadstr_f = strcat("..\pythonTest_d_grid", int2str(grid), "_r_real_cov.mat");
else
    loadstr = strcat("..\randomUE_grid", int2str(grid), "_real.mat");
    loadstr_f = strcat("..\pythonTest_d_grid", int2str(grid), "_r_real.mat");
end

load(loadstr)
load(loadstr_f)

% Locations
loc_offline = loc_offline'; % "Known" locations, used to interpolate/predict online sample locations
loc_online = loc_online'; % real locations of onlinesamples

filepath = strcat('..\outputs\d_grid', int2str(grid));
for layers = layer_configs
    n_layer = length(nonzeros(layers));
    filename = strcat('_',int2str(n_layer), 'layer');
    for layer = layers'
        if layer ~= 0
            filename = strcat(filename, '_');
            filename = strcat(filename, int2str(layer));
        end
    end
    if cov == 1
        filename = strcat(filename, '_real_cov', '.mat');
    else
        filename = strcat(filename, '_r_real', '.mat');
    end
    file = strcat(filepath, filename);
    
    load(file); % Load a specific layer file
     
    d_eucl = zeros([n_online n_online]);
    z_eucl = zeros([n_online n_online]);
    f_eucl = zeros([n_online n_online]);
    err_mat_z = zeros([n_online n_online]); % squared
    err_mat_f = zeros([n_online n_online]);

    for i = 1:n_online
        for j = 1:n_online
            z_eucl(i,j) = z_d((i-1)*n_online+j);
            d_eucl(i,j) = phys_d((i-1)*n_online+j);
            f_eucl(i,j) = norm(x_online(1,:,(i-1)*n_online+j)'-x_online(2,:,(i-1)*n_online+j)');

            err_mat_z(i,j) = (z_d((i-1)*n_online+j) - phys_d((i-1)*n_online+j))^2;
            err_mat_f(i,j) = (norm(x_online(1,:,(i-1)*n_online+j)'-x_online(2,:,(i-1)*n_online+j)') - phys_d((i-1)*n_online+j))^2;
        end
    end

    err_vec_z = zeros([(n_online-1)*n_online/2 1]);
    err_vec_f = zeros([(n_online-1)*n_online/2 1]);
    c = 1;
    for i = 1:n_online
        for j = i+1:n_online
            err_vec_z(c) = err_mat_z(i,j);
            err_vec_f(c) = err_mat_f(i,j);
            c = c + 1;
        end
    end
    
    RMSE_z = sqrt(mean(err_vec_z))
    RMSE_f = sqrt(mean(err_vec_f))

    pearMat_z = corrcoef(d_eucl(:), z_eucl(:));
    pearMat_f = corrcoef(d_eucl(:), f_eucl(:));
    corr_z = pearMat_z(2)
    corr_f = pearMat_f(2)
    % CDFS etc.

    % fprintf(fileID_z, '%s METRIC WKNN: pearson = %f, min RMSE = %fm, (k = %d), 80th percentile = %fm, 90th percentile = %fm, grid = %d, weight factor = %f\n', filename, pear_z, min_z, kk(i_z), val_80th_z, val_90th_z,grid, factor_z);
    % fprintf(fileID_z, '%s EUCLIDEAN WKNN: pearson = %f, min RMSE = %fm, (k = %d), 80th percentile = %fm, 90th percentile = %fm, grid = %d, weight factor = %f\n', filename, pear_f, min_f, kk(i_f), val_80th_f, val_90th_f,grid, factor_f);
end
fclose(fileID_z);
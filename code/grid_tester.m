clear
close all;

layer_configs = [
    [512,256,128,64,32,16]; % [256,128,32,16,8,4];
]';

% layer_configs = [[64,32,16,8 0 0]; [128,32,16,8 0 0]; [64,16,8,4 0 0]; [128,64,16,8 0 0]; [64,32,8,4 0 0]; [128,64,32,16 0 0]; % 4 layer configs
%     [2,16,8,2 0 0]; [64,9,18,128 0 0]; [4,32,2,32 0 0]; [4,4,2,32 0 0]; [2,4,8,16 0 0]; [64,64,2,16 0 0]; [32,16,8,4 0 0];
%     [32,16,8,2 0 0]; [32,16,8,2 0 0]; [16,8,4,2 0 0]; [64,16,8,2 0 0]; [32,8,4,2 0 0]; [64,32,16,2 0 0];
%     [64,64,32,16,4 0]; [128,64,32,16,8 0]; [128,32,16,8 ,4 0]; [256, 64,16,8,4 0]; [256,128,64,16,8 0]; [256,64,32,8,4 0]; [256,128,64,32,16 0]; % 5 layer configs
%     [256,128,64,32,16,8]; [256,128,32,16,8,4]; [256,64,32,16,8,4]; % 6 layer
%     [256,128,64,16,8,4]; [256,64,32,8,4,8]; [512,256,128,64,32,16];
% ]';

grid = 3;
k_bool = 1;
k_v = 32;



loadstr = strcat("..\randomUE_grid", int2str(grid), "_real.mat");
load(loadstr)

loadstr_f = strcat("..\pythonTest_grid", int2str(grid), "_r_real_cov.mat");
load(loadstr_f)

% Locations
loc_offline = loc_offline'; % "Known" locations, used to interpolate/predict online sample locations
loc_online = loc_online'; % real locations of onlinesamples

filepath = strcat('..\outputs\grid', int2str(grid));
for layers = layer_configs
    n_layer = length(nonzeros(layers));
    filename = strcat('_',int2str(n_layer), 'layer');
    for layer = layers'
        if layer ~= 0
            filename = strcat(filename, '_');
            filename = strcat(filename, int2str(layer));
        end
    end
    if k_bool == 1
        filename = strcat(filename, "_r_real_cov_k" , int2str(k_v) , ".mat");
    else
        filename = strcat(filename, '_real_cov', '.mat');
    end
    file = strcat(filepath, filename);
    
    load(file); % Load a specific layer file
     
    d_eucl = zeros([n_online n_offline]);
    z_eucl = zeros([n_online n_offline]);
    f_eucl = zeros([n_online n_offline]);

    for i = 1:n_online
        for j = 1:n_offline
            z_eucl(i,j) = z_d((i-1)*n_offline+j);
            d_eucl(i,j) = phys_d((i-1)*n_offline+j);
            f_eucl(i,j) = norm(x_online(1,:,(i-1)*n_offline+j)'-x_online(2,:,(i-1)*n_offline+j)');
        end
    end
    
    z_eucl = z_eucl/max(z_eucl(:));
    f_eucl = f_eucl/max(f_eucl(:));
    d_eucl = d_eucl/max(d_eucl(:));
    
    % dz_sort = zeros([n_online n_offline]);
    % df_sort = zeros([n_online n_offline]);
    % z_sort = zeros([n_online n_offline]);
    % f_sort = zeros([n_online n_offline]);
    % for row = 1:n_online
    %     [~,i_z] = sort(z_eucl(row,:));
    %     [~,i_f] = sort(f_eucl(row,:));
    % 
    %     z_sort(row,:) = z_eucl(row,i_z);
    %     f_sort(row,:) = f_eucl(row,i_f);
    % 
    %     dz_sort(row,:) = d_eucl(row,i_z);
    %     df_sort(row,:) = d_eucl(row,i_f);
    % end
    % 
    % %[test_z, i_z] = sort(z_eucl,2);
    % %d_sort = d_eucl(i_z);
    % %scatter(d_eucl(:), z_eucl(:))
    % %z_sort = z_eucl(i_z);
    % scatter(test_z(:), dz_sort(:))

    
    k = k_v;

    dz_sort = zeros([n_online k]);
    df_sort = zeros([n_online k]);
    z_sort = zeros([n_online k]);
    f_sort = zeros([n_online k]);
    for row = 1:n_online
        [~,i_z] = sort(z_eucl(row,:));
        [~,i_f] = sort(f_eucl(row,:));

        z_sort(row,:) = z_eucl(row,i_z(1:k));
        f_sort(row,:) = f_eucl(row,i_f(1:k));

        dz_sort(row,:) = d_eucl(row,i_z(1:k));
        df_sort(row,:) = d_eucl(row,i_f(1:k));
    end
    
    pearMat_z_all = corrcoef(d_eucl(:), z_eucl(:));
    pear_z_all = pearMat_z_all(2);

    pearMat_f_all = corrcoef(d_eucl(:), f_eucl(:));
    pear_f_all = pearMat_f_all(2);
    
    pearMat_z_k = corrcoef(dz_sort(:), z_sort(:));
    pear_z_k = pearMat_z_k(2);

    pearMat_f_k = corrcoef(df_sort(:), f_sort(:));
    pear_f_k = pearMat_f_k(2);

    disp("z_all: " + sprintf('%.6f',pear_z_all))
    disp("z_k: "+ sprintf('%.6f',pear_z_k))
    disp("f_all: "+ sprintf('%.6f',pear_f_all))
    disp("f_k: "+ sprintf('%.6f',pear_f_k))
    
    figure()
    scatter(df_sort(:), f_sort(:))
    ylim([0 1])
    xlim([0 1])
    figure()
    scatter(dz_sort(:), z_sort(:))
    ylim([0 1])
    xlim([0 1])
    %scatter(df_sort(:), f_sort(:))
    %scatter(d_eucl(:), z_eucl(:))
end

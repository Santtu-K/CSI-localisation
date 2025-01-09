clear
close all;

fileID_z = fopen("res_r_real2.text", "a");
fprintf(fileID_z,'\nTime: %s\n', datestr(datetime("now")));

fileID_f = fopen("res_r_real_feat2.text", "a");
fprintf(fileID_f,'\nTime: %s\n', datestr(datetime("now")));

layer_configs = [
   [512,256,128,64,32,16]; %  [256,128,32,16,8,4]; 
]';

% layer_configs = [[64,32,16,8 0 0]; [128,32,16,8 0 0]; [64,16,8,4 0 0]; [128,64,16,8 0 0]; [64,32,8,4 0 0]; [128,64,32,16 0 0]; % 4 layer configs
%     [2,16,8,2 0 0]; [64,9,18,128 0 0]; [4,32,2,32 0 0]; [4,4,2,32 0 0]; [2,4,8,16 0 0]; [64,64,2,16 0 0]; [32,16,8,4 0 0];
%     [32,16,8,2 0 0]; [32,16,8,2 0 0]; [16,8,4,2 0 0]; [64,16,8,2 0 0]; [32,8,4,2 0 0]; [64,32,16,2 0 0];
%     [64,64,32,16,4 0]; [128,64,32,16,8 0]; [128,32,16,8 ,4 0]; [256, 64,16,8,4 0]; [256,128,64,16,8 0]; [256,64,32,8,4 0]; [256,128,64,32,16 0]; % 5 layer configs
%     [256,128,64,32,16,8]; [256,128,32,16,8,4]; [256,64,32,16,8,4]; % 6 layer
%     [256,128,64,16,8,4]; [256,64,32,8,4,8]; [512,256,128,64,32,16];
% ]';

grid = 3;

loadstr = strcat("..\randomUE_grid", int2str(grid), "_real.mat");
load(loadstr)

loadstr_f = strcat("..\pythonTest_grid", int2str(grid), "_r_real.mat");
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
    filename = strcat(filename, '_real', '.mat');
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

    
    
    n_round = 1;
    kk = [];

    for i = 2:20
        if i > 20
            break
        end
        kk = [kk, i]; % k in k-NN 
    end
    kk = 4;

    rmses_z = zeros([n_round length(kk)]); % WKNN
    rmses_comp_z = zeros([n_round length(kk)]); % KNN
    tot_errors_z = zeros([n_online n_round length(kk)]);
    tot_errors_comp_z = zeros([n_online n_round length(kk)]);

    KNN_tally_z = 0;
    WKNN_tally_z = 0;

    rmses_f = zeros([n_round length(kk)]); % WKNN
    rmses_comp_f = zeros([n_round length(kk)]); % KNN
    tot_errors_f = zeros([n_online n_round length(kk)]);
    tot_errors_comp_f = zeros([n_online n_round length(kk)]);

    KNN_tally_f = 0;
    WKNN_tally_f = 0;

    for i_out = 1:n_round
        z_eucl; % Pairwise feature distances between
                % online and offline samples.  
        f_eucl; % Pairwise feature distances between
                % online and offline samples. 

        ind = 0;
        for k = kk

            ind = ind + 1;
            errors_z = zeros([n_online 2]); % Raw error vector for x and y (WKNN)
            errors_comp_z = zeros([n_online 2]); % Raw error vector for x and y (KNN)

            errors_f = zeros([n_online 2]); % Raw error vector for x and y (WKNN)
            errors_comp_f = zeros([n_online 2]); % Raw error vector for x and y (KNN)

            for i = 1:n_online
               
                test_sample_z = z_eucl(i,:); % Current test sample (the feature distances between test/new sample and training samples)
                [test_sample_z_sort, i_sort_z] = sort(test_sample_z, "ascend"); % Sorted by nearness

                test_sample_f = f_eucl(i,:); % Current test sample (the feature distances between test/new sample and training samples)
                [test_sample_f_sort, i_sort_f] = sort(test_sample_f, "ascend"); % Sorted by nearness
           
                k_NN_z = i_sort_z(1:k); % k-NN
                k_NN_f = i_sort_f(1:k); % k-NN
                
                KNN_z = loc_offline(k_NN_z,:);
                KNN_z_d = test_sample_z_sort(1:k); % for calculating the weights

                KNN_f = loc_offline(k_NN_f,:);
                KNN_f_d = test_sample_f_sort(1:k); % for calculating the weights
                
                factor_z = 1/2; % How the weights are calculated 
                w_z = exp(KNN_z_d*(-1/factor_z))'; % Closer in terms of feature (z) distance --> larger weight

                factor_f = 1/2; % How the weights are calculated 
                w_f = exp(KNN_f_d*(-1/factor_f))'; % Closer in terms of feature (z) distance --> larger weight
                
                WKNN_z = KNN_z.*w_z;
                w_sum_z = sum(w_z);

                WKNN_f = KNN_f.*w_f;
                w_sum_f = sum(w_f);

                pos_est_z = sum(WKNN_z/w_sum_z);
                pos_est_f = sum(WKNN_f/w_sum_f);
                pos_true = loc_online(i,:); % True position

                KNN_res_z = mean(loc_offline(k_NN_z,:));
                KNN_res_f = mean(loc_offline(k_NN_f,:));
                
                magic_num = round(n_online*length(kk)*1/10);
                r = randi(magic_num);
                if r == randi(magic_num)
                    "hit"
                    figure()
                    hold on
                    scatter(KNN_z(:,1), KNN_z(:,2),"o")
                    scatter(KNN_f(:,1), KNN_f(:,2),"square")
                    scatter(pos_est_z(1), pos_est_z(2), "x") % Estimation X/z
                    scatter(pos_est_f(1), pos_est_f(2), "*") % Estimation */f
                    scatter(pos_true(1), pos_true(2), "d") % True O
                    %scatter(KNN_res_z(1), KNN_res_z(2),"+")
                    ylim([-10 10])
                    xlim([-10 10])
                end
            
                W_KNN_err_z = abs(pos_est_z - pos_true); % Raw error for x and y (WKNN)
                KNN_err_z = abs(KNN_res_z - pos_true); % Raw error for x and y (KNN)

                W_KNN_err_f = abs(pos_est_f - pos_true); % Raw error for x and y (WKNN)
                KNN_err_f = abs(KNN_res_f - pos_true); % Raw error for x and y (KNN)

                if W_KNN_err_z > KNN_err_z
                    KNN_tally_z = KNN_tally_z + 1;
                else
                    WKNN_tally_z = WKNN_tally_z + 1;
                end

                if W_KNN_err_f > KNN_err_f
                    KNN_tally_f = KNN_tally_f + 1;
                else
                    WKNN_tally_f = WKNN_tally_f + 1;
                end
                
                errors_comp_z(i,:) = KNN_err_z;
                errors_z(i,:) = W_KNN_err_z;

                errors_comp_f(i,:) = KNN_err_f;
                errors_f(i,:) = W_KNN_err_f;
            end
            
            % Location errors for a given k, z
            tot_error_z = sqrt(sum(errors_z.^2,2));
            tot_error_comp_z = sqrt(sum(errors_comp_z.^2,2));
            
            rmse_comp_z = sqrt(mean(tot_error_comp_z.^2));
            rmse_z = sqrt(mean(tot_error_z.^2));

            rmses_comp_z(i_out, ind) = rmse_comp_z;
            rmses_z(i_out, ind) = rmse_z;

            tot_errors_z(:,i_out, ind) = tot_error_z;
            tot_errors_comp_z(:,i_out, ind) = tot_error_comp_z;

            % Location errors for a given k, f
            tot_error_f = sqrt(sum(errors_f.^2,2));
            tot_error_comp_f = sqrt(sum(errors_comp_f.^2,2));
            
            rmse_comp_f = sqrt(mean(tot_error_comp_f.^2));
            rmse_f = sqrt(mean(tot_error_f.^2));

            rmses_comp_f(i_out, ind) = rmse_comp_f;
            rmses_f(i_out, ind) = rmse_f;

            tot_errors_f(:,i_out, ind) = tot_error_f;
            tot_errors_comp_f(:,i_out, ind) = tot_error_comp_f;
        end
    end
    
    % avg_tot_error_z = mean(tot_errors_z);
    % avg_tot_error_comp = mean(tot_errors_comp);

    [min_z, i_z] = min(rmses_z);
    [min_comp_z, i_comp_z] = min(rmses_comp_z);
    [min_f, i_f] = min(rmses_f);
    [min_comp_f, i_comp_f] = min(rmses_comp_f);
    
    % figure()
    % scatter(kk, avg_rmse_z, 100, "x")
    % xlabel("k")
    % ylabel("RMSE")
    % title("RMSE versus k")

    pearMat_z = corrcoef(d_eucl(:), z_eucl(:));
    pear_z = pearMat_z(2);

    pearMat_f = corrcoef(d_eucl(:), f_eucl(:));
    pear_f = pearMat_f(2);
    
    % z distance
    [d_CDF_z, d_edge_z]= histcounts(tot_errors_z(:,:,i_z), 100, 'Normalization',  'cdf');
    [d_CDF_comp_z, d_edge_comp_z]= histcounts(tot_errors_comp_z(:,:,i_comp_z), 100, 'Normalization',  'cdf');
    size_ed = length(d_edge_z);
    th80perc = zeros([size_ed 1]) + 0.8;
    th90perc = zeros([size_ed 1]) + 0.9;

    [cdf_val_z_80, cdf_i_z_80] = max(d_CDF_z >= 0.8);
    val_80th_z = d_edge_z(cdf_i_z_80);

    [cdf_val_z_90, cdf_i_z_90] = max(d_CDF_z >= 0.9);
    val_90th_z = d_edge_z(cdf_i_z_90);

    [cdf_val_comp_z, cdf_i_comp_z] = max(d_CDF_comp_z >= 0.8);
    val_80th_comp_z = d_edge_comp_z(cdf_i_comp_z);
    
    % feature distance
    [d_CDF_f, d_edge_f]= histcounts(tot_errors_f(:,:,i_f), 100, 'Normalization',  'cdf');
    [d_CDF_comp_f, d_edge_comp_f]= histcounts(tot_errors_comp_f(:,:,i_comp_f), 100, 'Normalization',  'cdf');
    size_ed = length(d_edge_f);
    th80perc = zeros([size_ed 1]) + 0.8;

    [cdf_val_f_80, cdf_i_f_80] = max(d_CDF_f >= 0.8);
    val_80th_f = d_edge_f(cdf_i_f_80);

    [cdf_val_f_90, cdf_i_f_90] = max(d_CDF_f >= 0.9);
    val_90th_f = d_edge_f(cdf_i_f_90);

    [cdf_val_comp_f, cdf_i_comp_f] = max(d_CDF_comp_f >= 0.8);
    val_80th_comp_f = d_edge_comp_f(cdf_i_comp_f);

    % figure
    % hold on
    % plot(d_edge_z,[0,d_CDF_z], 'lineWidth', 2)
    % plot(d_edge_z, th80perc, ".")
    % plot(d_edge_z, th90perc, ".")
    % title("Metric learning localization error CDF")
    % xlabel("Localization error (m)")
    % ylabel("CDF")
    % ylim([0 1])
    % hold off

    figure
    hold on
    plot(d_edge_f,[0,d_CDF_f], 'lineWidth', 2)
    plot(d_edge_z,[0,d_CDF_z], 'lineWidth', 2)
    plot(d_edge_f, th80perc, ".")
    plot(d_edge_f, th90perc, ".")
    title("localization error CDF feat")
    xlabel("Localization error (m)")
    ylabel("CDF")
    legend("feature", "metric")
    ylim([0 1])
    hold off
    
    ratio_z = WKNN_tally_z / KNN_tally_z;
    ratio_f = WKNN_tally_f / KNN_tally_f;
    
    % % Determining the best weighting scheme for WKNN
    % ratio_read = fopen("bestFactor_r.text", "r"); % reading
    % line1 = fgetl(ratio_read);
    % line2 = fgetl(ratio_read);
    % parts = split(line1, '= ');
    % ratio_prev = parts(3);
    % t1 = split(ratio_prev, ',');
    % ratio_prev = str2double(t1(1));
    % fclose(ratio_read);
    % 
    % ratio_write = fopen("bestFactor_r.text", "w"); % writing
    % if ratio > ratio_prev
    %     fprintf(ratio_write, 'Best factor thus far: %s: ratio = %d/%d = %f, n = %d, factor = %f \n', filename, WKNN_tally, KNN_tally, ratio, n_samp, factor);
    %     temp = split(line1, "Best factor thus far: ");
    %     fprintf(ratio_write, 'Previous best factor: %s', char(strtrim(temp(2))));
    %     fprintf(ratio_write, '\n\nMost recent run:      %s: ratio = %d/%d = %f, n = %d, factor = %f', filename, WKNN_tally, KNN_tally, ratio, n_samp, factor);
    % else
    %     fprintf(ratio_write, '%s\n', line1);
    %     fprintf(ratio_write, '%s\n', line2);
    %     fprintf(ratio_write, '\nMost recent run:      %s: ratio = %d/%d = %f, n = %d, factor = %f', filename, WKNN_tally, KNN_tally, ratio, n_samp, factor);
    % end    
    % fclose(ratio_write);

    fprintf(fileID_z, '%s METRIC WKNN: pearson = %f, min RMSE = %fm, (k = %d), 80th percentile = %fm, 90th percentile = %fm, grid = %d, weight factor = %f\n', filename, pear_z, min_z, kk(i_z), val_80th_z, val_90th_z,grid, factor_z);
    %fprintf(fileID_z, '%s KNN: pearson = %f, min RMSE = %fm, (k = %d), 80th percentile = %fm, grid = %d\n', filename, pear_z, min_comp_z, kk(i_comp_z), val_80th_comp_z,grid);

    fprintf(fileID_z, '%s EUCLIDEAN WKNN: pearson = %f, min RMSE = %fm, (k = %d), 80th percentile = %fm, 90th percentile = %fm, grid = %d, weight factor = %f\n', filename, pear_f, min_f, kk(i_f), val_80th_f, val_90th_f,grid, factor_f);
    %fprintf(fileID_f, '%s KNN: pearson = %f, min RMSE = %fm, (k = %d), 80th percentile = %fm, grid = %d\n', filename, pear_f, min_comp_f, kk(i_comp_f), val_80th_comp_f,grid);
end
fclose(fileID_z);
fclose(fileID_f);

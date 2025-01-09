% You may find the usage example of the functions in Matlab Documentation

% More info about cell structure in Matlab:
% https://se.mathworks.com/help/matlab/ref/cell.html

% you may check the usage in Matlab help center:
% scatter()
% squeeze()
% diag()

clear % clear the values in Worksapce

load('cov_4ant_2bs.mat') 

% ###### a correction for the BS coordinates! #####
BS_position = BS_position(:,[1,4]); % we consider 2 BSs in the work
% ###### a correction for the BS coordinates! #####

% plot the BS and UE locations
% This is an example to use scatter function in Matlab
figure % create a figure
hold on % hold on is used for two scatter plots in one figure
scatter(BS_position(1,:), BS_position(2,:), 'filled','^')
scatter(UE_position_all(1,:), UE_position_all(2,:),'.')
legend('BS','UE') % add legend for two scatter polts
hold off

% Parameters in the dataset
N_UE  = size(UE_position_all, 2); % number of UEs
N_BS  = size(R_Cov,1);            % number of BSs
N_Ant = size(R_Cov{1,1},2);       % number of antennas for each BS

% create storage space for features
cov      = zeros(N_BS*N_Ant*N_Ant,N_UE); % covariances
cov_log  = zeros(N_BS*N_Ant*N_Ant,N_UE); % log scale covariances
cov_temp = zeros(N_UE,N_Ant,N_Ant);

% Loops to save covariances in the vector of real values
% 1. Firstly we save 4 diagonal values. they are real values.
% 2. Secondly we have a loop to save the i_index-th diagonal values. They are
% complex values, and we extracte the real and imag parts by functions real() and imag(). 
for i_u = 1:N_UE
    diag_save = [];
    diag_log_save = [];
    for i_b = 1:N_BS
        R = squeeze(R_Cov{i_b,1}(i_u,:,:));
        R_log = squeeze(R_Cov_log{i_b,1}(i_u,:,:));
        diag_save = [diag_save; diag(R)]; % save 4 diagonal values
        diag_log_save = [diag_log_save; diag(R_log)];
        for i_index = 1: N_Ant-1 % save the i_index-th diagonal values
            temp_diag = diag(R,i_index);
            diag_save = [diag_save; real(temp_diag); imag(temp_diag)];
            
            temp_log_diag = diag(R_log,i_index);
            diag_log_save = [diag_log_save; real(temp_log_diag); imag(temp_log_diag)];
        end
    end
    cov(:,i_u) = diag_save; % save input feature for each UE
    cov_log(:,i_u) = diag_log_save;
end

% Finally we save the input for neural networks
save MLP_input_cov_log cov_log UE_position_all  -v7.3
% save MLP_input cov_log UE_position_all -v7.3

clear 
close all;

load('..\MLP_input_log.mat') % log cov data
n_ue = length(cov_log); % cov or cov_log

% load('..\MLP_input_cov.mat') % log cov data
% n_ue = length(cov); % cov or cov_log

% n = 2116; % number of total UE
% n_offline =  1000; % Training UE amount
% n_online = n-n_offline; % Testing UE amount

% ind_all = randperm(n_ue);
% i_offline = ind_all(1:n_offline); % Choose n_offline UEs for training
% i_online = ind_all(n_offline+1:n_offline+n_online); % Choose n_online UEs for testing

grid =3;

ind_all = 1:n_ue;
i_offline = 1:grid:n_ue; % Choose n_offline UEs for training
i_offline(mod(i_offline, 46*grid) > 46) = [];
i_online = ind_all; i_online(i_offline) = [];%ind_all(n_offline+1:n_offline+n_online); % Choose n_online UEs for testing

% n_i_online = length(i_online);
 n_test = 300;
% 
% ind_test = randperm(n_i_online);
% i_test = ind_test(1:n_test);
% i_test = i_online(i_test);

%save('randomTestPoints.mat',"i_test", "-v7.3")
load('randomTestPoints.mat') % currently n_test = 300

n = 2116; % number of total UE
n_offline =  length(i_offline); % Training UE amount
n_online = n_test; % Testing UE amount

% Offline (training) samples
f_offline = zeros([32 n_offline]); % feature
loc_offline = zeros([2 n_offline]); % phys coordinates

f_online = zeros([32 n_online]); % feature
loc_online = zeros([2 n_online]); % phys coordinates

loc_offline(:,1:n_offline) = UE_position_all(:,i_offline);
loc_online(:,1:n_online) = UE_position_all(:,i_test);

% Online (testing) samples

if exist('cov') == 1
    f_online(:,1:n_online) = cov(:,i_test);
    f_offline(:,1:n_offline) = cov(:,i_offline);
else
    f_online(:,1:n_online) = cov_log(:,i_test);
    f_offline(:,1:n_offline) = cov_log(:,i_offline);
end

y_mat_offline = zeros([n_offline n_offline]); % real physical distance between UEs
x_mat_offline = zeros([2 32 n_offline n_offline]); % corresponding UE features

for i = 1:n_offline
    feat1 = f_offline(:,i);
    pos1 = loc_offline(:,i);
    for j = 1:n_offline
        feat2 = f_offline(:,j);
        pos2 = loc_offline(:,j);

        y_mat_offline(i,j) = norm(pos1-pos2); % d_ij
        x_mat_offline(:,:,i,j) = [feat1, feat2]';
    end  
end


% t_sort = zeros([n_offline k]);
% y_sort = zeros([n_offline k]);
% x_sort = zeros([2 32 n_offline k]);%zeros([n_offline k 32 2]);

% for row = 1:n_offline
%     [~,i_t] = sort(t_mat(row,:));
%     [~,i_d] = sort(y_mat_offline(row,:));
% 
%     y_sort(row,:) = y_mat_offline(row,i_t(1:k));
%     x_sort(:,:,row,:) = x_mat_offline(:,:,row,i_t(1:k));
% end


y_mat_online = zeros([n_online n_online]); % real physical distance between UEs
x_mat_online = zeros([2 32 n_offline n_online]); % corresponding UE features
t_mat = zeros([n_online n_online]); % feat distance between UEs
err_mat = zeros([n_online n_online]); % squared

for i = 1:n_online
    feat1 = f_online(:,i); 
    pos1 = loc_online(:,i);
    for j = 1:n_online
        feat2 = f_online(:,j);
        pos2 = loc_online(:,j);

        y_mat_online(j,i) = norm(pos1-pos2);
        t_mat(i,j) = norm(feat1-feat2); % fd_ij
        err_mat(i,j) = (norm(pos1-pos2) - norm(feat1-feat2))^2;
        x_mat_online(:,:,j,i) = [feat1, feat2]';
    end
end

err_vec = zeros([(n_online-1)*n_online/2 1]);
c = 1;
for i = 1:n_online
    for j = i+1:n_online
        err_vec(c) = err_mat(i,j);
        c = c + 1;
    end
end

RMSE = sqrt(mean(err_vec))

% x_offline = x_mat_offline(:,:,:);
% y_offline = y_mat_offline(:)';

x_offline = x_mat_offline(:,:,:);
y_offline = y_mat_offline(:)';


x_online = x_mat_online(:,:,:);
y_online = y_mat_online(:)';

% tester = zeros([1 length(y_online)]);
% for i = 1:length(y_online)
%     tester(:,i) = norm(x_online(1,:,i)-x_online(2,:,i));
% end
% figure()
% scatter(tester(:)/max(tester(:)), y_online(:)/max(y_online(:)))

figure(1)
hold on 
scatter([-10 10], [10 -10], 100, "*")
scatter(loc_offline(1,:), loc_offline(2,:), "+") 
scatter(loc_online(1,:), loc_online(2,:), "d") 
legend("BS", "training", "testing")
ylim([-10 10])
xlim([-10 10])

str_train = strcat("..\pythonTrain_d_grid", int2str(grid), "_r_real", ".mat");
str_test = strcat("..\pythonTest_d_grid", int2str(grid), "_r_real", ".mat");
str2 = strcat("..\randomUE_grid", int2str(grid), "_real.mat");

if exist('cov') == 1
    str_train = strcat("..\pythonTrain_d_grid", int2str(grid), "_r_real_cov_k", ".mat");
    str_test = strcat("..\pythonTest_d_grid", int2str(grid), "_r_real_cov", ".mat");
    str2 = strcat("..\randomUE_d_grid", int2str(grid), "_real_cov.mat");
end

% % dataset for pytorch ML/NN, training
save(str_train,"x_offline", "y_offline", "-v7.3");

% dataset for pytorch ML/NN, testing
save(str_test,"x_online", "y_online", "-v7.3");

% Save the random n UE positions for WKNN/KNN
save(str2,"loc_online", "loc_offline", "ind_all", "i_offline", "i_online", "n_offline", "n_online", "-v7.3")
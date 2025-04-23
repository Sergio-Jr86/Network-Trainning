clear; clc; close all;

% --- Seleção da pasta com arquivos CSV ---
folder_path = uigetdir('', 'Selecione a pasta com os arquivos CSV');
if folder_path == 0
    error('Nenhuma pasta selecionada. O script será encerrado.');
end

% --- Listar arquivos CSV na pasta ---
file_list = dir(fullfile(folder_path, '*.csv'));
arquivos = {file_list.name};

% --- Configurações iniciais ---
cutoff_freq = 10;
inputs = [];
outputs = [];

% Filtro passa-baixa (100 Hz)
[b, a] = butter(4, cutoff_freq / (100 / 2), 'low');

% Colunas necessárias
required_columns = {'P6_RF_acc_x', 'P6_RF_acc_y', 'P6_RF_acc_z', ...
                    'rightTotalForce_N_'};

valid_files = {};

for i = 1:length(arquivos)
    arquivo = fullfile(folder_path, arquivos{i});
    dados = readtable(arquivo);

    if all(ismember(required_columns, dados.Properties.VariableNames))
        valid_files{end+1} = arquivo;

        acc_x = dados.P6_RS_acc_x;
        acc_y = dados.P6_RS_acc_y;
        acc_z = dados.P6_RS_acc_z;
        grf_right = dados.rightTotalForce_N_;

        acc_x = filtfilt(b, a, acc_x);
        acc_y = filtfilt(b, a, acc_y);
        acc_z = filtfilt(b, a, acc_z);

        acc_x = (acc_x - mean(acc_x)) / std(acc_x);
        acc_y = (acc_y - mean(acc_y)) / std(acc_y);
        acc_z = (acc_z - mean(acc_z)) / std(acc_z);

        inputs_atual = [acc_x, acc_y, acc_z];
        outputs_atual = grf_right;

        inputs = [inputs; inputs_atual];
        outputs = [outputs; outputs_atual];
    end
end

if isempty(valid_files)
    error('Nenhum arquivo válido foi encontrado na pasta selecionada.');
end

disp(['Arquivos utilizados no treinamento: ', num2str(length(valid_files))]);

% Normalização da saída
mean_grf = mean(outputs, 1);
std_grf = std(outputs, 0, 1);
outputs = (outputs - mean_grf) ./ std_grf;
save('normalization_params.mat', 'mean_grf', 'std_grf');

% --- Criação de janelas temporais ---
time_window = 30;
step_sizes = [1, 3, 5];

X = {};
Y = [];
for step_size = step_sizes
    for i = 1:step_size:(size(inputs, 1) - time_window)
        janela = inputs(i:i + time_window - 1, :);
        X{end+1} = janela';
        Y = [Y; outputs(i + time_window - 1, :)];
    end
end

disp(['Dimensão final de X: ', mat2str(size(X))]);
disp(['Dimensão final de Y: ', mat2str(size(Y))]);

% --- K-Fold ---
k = 5;
indices = crossvalind('Kfold', size(Y, 1), k);

rmse_folds = zeros(k, 1);
rRMSE_folds = zeros(k, 1);
r2_folds = zeros(k, 1);

for fold = 1:k
    disp(['Iniciando fold ', num2str(fold), ' de ', num2str(k)]);
    test_idx = (indices == fold);
    train_idx = ~test_idx;

    X_train = X(train_idx);
    Y_train = Y(train_idx, :);
    X_test = X(test_idx);
    Y_test = Y(test_idx, :);

    if isempty(X_train) || isempty(Y_train)
        error('Os dados de treinamento estão vazios.');
    end

    % --- Arquitetura TCN ---
    layers = [
        sequenceInputLayer(size(X{1}, 1))
        convolution1dLayer(3, 128, 'Padding', 'causal', 'DilationFactor', 1)
        reluLayer
        dropoutLayer(0.3)
        convolution1dLayer(3, 64, 'Padding', 'causal', 'DilationFactor', 2)
        reluLayer
        dropoutLayer(0.3)
        globalMaxPooling1dLayer
        fullyConnectedLayer(64)
        reluLayer
        fullyConnectedLayer(1)
        regressionLayer
    ];

    options = trainingOptions('adam', ...
        'MaxEpochs', 30, ...
        'InitialLearnRate', 5e-3, ...
        'LearnRateSchedule', 'piecewise', ...
        'LearnRateDropFactor', 0.8, ...
        'LearnRateDropPeriod', 10, ...
        'MiniBatchSize', 128, ...
        'Shuffle', 'every-epoch', ...
        'ValidationData', {X_test, Y_test}, ...
        'ValidationFrequency', 50, ...
        'ValidationPatience', 10, ...
        'Verbose', false, ...
        'Plots', 'training-progress', ...
        'L2Regularization', 1e-4, ...
        'GradientThreshold', 1);

    if isempty(X_test) || isempty(Y_test)
        error('Os dados de teste estão vazios.');
    end

    % --- Treinamento ---
    net = trainNetwork(X_train, Y_train, layers, options);

    % --- Previsão e dessnormalização ---
    predictions = predict(net, X_test);

    load('normalization_params.mat', 'mean_grf', 'std_grf');
    predictions = (predictions .* std_grf) + mean_grf;
    Y_test = (Y_test .* std_grf) + mean_grf;

    % --- Métricas ---
    rmse_folds(fold) = sqrt(mean((Y_test - predictions).^2));
    range_val = max(Y_test) - min(Y_test);
    rRMSE_folds(fold) = (rmse_folds(fold) / range_val) * 100;

    ss_total = sum((Y_test - mean(Y_test)).^2);
    ss_residual = sum((Y_test - predictions).^2);
    r2_folds(fold) = 1 - (ss_residual / ss_total);

    % --- Exportar resíduos ---
    residuos = Y_test - predictions;
    residuos_tabela = table((1:length(Y_test))', Y_test, predictions, residuos, ...
        'VariableNames', {'Sample', 'GRF_real', 'GRF_predito', 'Resíduo'});
    nome_arquivo_residuos = sprintf('residuos_TCN_fold_%d.csv', fold);
    writetable(residuos_tabela, nome_arquivo_residuos);

    disp(['Resíduos do fold ', num2str(fold), ' salvos em "', nome_arquivo_residuos, '".']);
end

% --- Resultados Finais ---
mean_rmse = mean(rmse_folds);
mean_rRMSE = mean(rRMSE_folds);
mean_r2 = mean(r2_folds);

disp('--- Resultados Finais ---');
disp(['Média RMSE: ', sprintf('%.2f', mean_rmse)]);
disp(['Média rRMSE (%): ', sprintf('%.2f', mean_rRMSE), '%']);
disp(['Média R²: ', sprintf('%.2f', mean_r2)]);

metricas_tabela = table((1:k)', rmse_folds, rRMSE_folds, r2_folds, ...
    'VariableNames', {'Fold', 'RMSE', 'rRMSE', 'R2'});
writetable(metricas_tabela, 'metricas_TCN_vGRF_RS.csv');
disp('Métricas salvas no arquivo "metricas_TCN_vGRF_RS.csv".');

save('modelo_TCN_vGRF_RS.mat', 'net');
disp('Modelo treinado salvo como "modelo_TCN_vGRF_RS.mat".');

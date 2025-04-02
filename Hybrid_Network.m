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
cutoff_freq = 10; % Frequência de corte do filtro passa-baixa em Hz
inputs = [];
outputs = [];

% Ajustar filtro passa-baixa para 100 Hz
[b, a] = butter(4, cutoff_freq / (100 / 2), 'low');

% Colunas necessárias
required_columns = {'P6_RF_acc_x', 'P6_RF_acc_y', 'P6_RF_acc_z', ...
                    'rightTotalForce_N_'};

valid_files = {}; % Armazena os arquivos que possuem todas as colunas

for i = 1:length(arquivos)
    arquivo = fullfile(folder_path, arquivos{i});
    dados = readtable(arquivo);

    % Verificar se todas as colunas necessárias estão presentes
    if all(ismember(required_columns, dados.Properties.VariableNames))
        valid_files{end+1} = arquivo;
        
        % Extrair dados de entrada (acelerômetro) e saída (vGRF)
        acc_x = dados.P6_RS_acc_x;
        acc_y = dados.P6_RS_acc_y;
        acc_z = dados.P6_RS_acc_z;
        grf_right = dados.rightTotalForce_N_;

        % --- Filtragem dos sinais de acelerômetro ---
        acc_x = filtfilt(b, a, acc_x);
        acc_y = filtfilt(b, a, acc_y);
        acc_z = filtfilt(b, a, acc_z);

        % --- Normalização Z-score ---
        acc_x = (acc_x - mean(acc_x)) / std(acc_x);
        acc_y = (acc_y - mean(acc_y)) / std(acc_y);
        acc_z = (acc_z - mean(acc_z)) / std(acc_z);

        % Concatenar as entradas e saídas
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

% Normalização global das saídas (vGRF)
mean_grf = mean(outputs, 1);
std_grf = std(outputs, 0, 1);
outputs = (outputs - mean_grf) ./ std_grf; % Normalizar vGRF
save('normalization_params.mat', 'mean_grf', 'std_grf');

% --- Criação de Janelas Temporais ---
time_window = 30;
step_sizes = [1, 3, 5];

X = {};
Y = [];
for step_size = step_sizes
    for i = 1:step_size:(size(inputs, 1) - time_window)
        janela = inputs(i:i + time_window - 1, :);
        X{end+1} = janela'; % Cada elemento agora tem dimensão (num_features, time_window)
        Y = [Y; outputs(i + time_window - 1, :)];
    end
end

disp(['Dimensão final de X: ', mat2str(size(X))]);
disp(['Dimensão final de Y: ', mat2str(size(Y))]);

% --- Validação Cruzada (K-Fold) ---
k = 5; % Número de folds
indices = crossvalind('Kfold', size(Y, 1), k);

% Inicializar vetores para métricas
rmse_folds = zeros(k, 1);
rRMSE_folds = zeros(k, 1);
r2_folds = zeros(k, 1);

for fold = 1:k
    disp(['Iniciando fold ', num2str(fold), ' de ', num2str(k)]);
    test_idx = (indices == fold);
    train_idx = ~test_idx;

    % Dados de treinamento e teste
    X_train = X(train_idx);
    Y_train = Y(train_idx, :);
    X_test = X(test_idx);
    Y_test = Y(test_idx, :);

    if isempty(X_train) || isempty(Y_train)
        error('Os dados de treinamento estão vazios. Verifique se há dados suficientes para cada fold.');
    end

    % --- Configuração da Rede Híbrida Bi-LSTM + TCN ---
    layers = [
        sequenceInputLayer(size(X{1}, 1))
        % Camadas Convolucionais (TCN)
        convolution1dLayer(3, 64, 'Padding', 'causal', 'DilationFactor', 1)
        batchNormalizationLayer
        reluLayer
        convolution1dLayer(3, 128, 'Padding', 'causal', 'DilationFactor', 2)
        batchNormalizationLayer
        reluLayer
        
        % Camadas Bi-LSTM
        bilstmLayer(128, 'OutputMode', 'sequence')
        dropoutLayer(0.4)
        bilstmLayer(64, 'OutputMode', 'last')
        dropoutLayer(0.4)
        
        % Camadas densas finais
        fullyConnectedLayer(64)
        reluLayer
        fullyConnectedLayer(1)
        regressionLayer
    ]

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
        error('Os dados de teste estão vazios. Verifique a configuração de validação cruzada.');
    end

    % Treinamento
    net = trainNetwork(X_train, Y_train, layers, options);

    % Previsão
    predictions = predict(net, X_test);

    % Calcular RMSE
    rmse_folds(fold) = sqrt(mean((Y_test - predictions).^2));

    % Calcular rRMSE
    range_val = max(Y_test) - min(Y_test);
    rRMSE_folds(fold) = (rmse_folds(fold) / range_val) * 100;

    % Calcular R²
    ss_total = sum((Y_test - mean(Y_test)).^2);
    ss_residual = sum((Y_test - predictions).^2);
    r2_folds(fold) = 1 - (ss_residual / ss_total);
% --- Calcular e salvar resíduos por fold ---
residuos = Y_test - predictions;

% Salvar resíduos e predições em arquivo CSV
residuos_tabela = table((1:length(Y_test))', Y_test, predictions, residuos, ...
    'VariableNames', {'Sample', 'GRF_real', 'GRF_predito', 'Resíduo'});

nome_arquivo_residuos = sprintf('residuos_Hibrido_fold_%d.csv', fold);
writetable(residuos_tabela, nome_arquivo_residuos);

disp(['Resíduos do fold ', num2str(fold), ' salvos em "', nome_arquivo_residuos, '".']);


end
% Calcular médias das métricas para o modelo híbrido
mean_rmse = mean(rmse_folds);
mean_rRMSE = mean(rRMSE_folds);
mean_r2 = mean(r2_folds);

% Exibir os resultados no console
disp('--- Resultados Finais da Rede Híbrida Bi-LSTM + TCN ---');
disp(['Média RMSE: ', sprintf('%.4f', mean_rmse)]);
disp(['Média rRMSE (%): ', sprintf('%.2f', mean_rRMSE), '%']);
disp(['Média R²: ', sprintf('%.4f', mean_r2)]);

% Criar uma tabela para salvar as métricas
metricas_tabela = table((1:k)', rmse_folds, rRMSE_folds, r2_folds, ...
    'VariableNames', {'Fold', 'RMSE', 'rRMSE', 'R2'});
writetable(metricas_tabela, 'metricas_Hibrida_BiLSTM_TCN.csv');

disp('Métricas salvas no arquivo "metricas_Hibrida_BiLSTM_TCN.csv".');

% --- Salvar Modelo Treinado ---
save('modelo_Hibrido_BiLSTM_TCN.mat', 'net');
disp('Modelo treinado salvo como "modelo_Hibrido_BiLSTM_TCN.mat".');

clear; clc; close all;

% --- Seleção do arquivo CSV para predição ---
[file, path] = uigetfile('*.csv', 'Selecione um arquivo CSV para predição');
if file == 0
    error('Nenhum arquivo selecionado. O script será encerrado.');
end
arquivo = fullfile(path, file);

% --- Carregar modelos e parâmetros de normalização ---
models = {'modelo_vGRF_RS.mat', 'modelo_TCN_vGRF_RS.mat', 'modelo_Hibrido_BiLSTM_TCN.mat'};
model_names = {'Bi-LSTM', 'TCN', 'Hybrid'};

predicoes = cell(1, length(models));
metricas = zeros(length(models), 3);

load('normalization_params.mat', 'mean_grf', 'std_grf');

% --- Configuração do filtro ---
cutoff_freq = 15; sampling_freq = 100;
[b, a] = butter(4, cutoff_freq / (sampling_freq / 2), 'low');

% --- Colunas necessárias ---
required_columns = {'P6_RS_acc_x', 'P6_RS_acc_y', 'P6_RS_acc_z'};
output_column = 'rightTotalForce_N_';

% --- Ler e validar dados ---
dados = readtable(arquivo);
if ~all(ismember(required_columns, dados.Properties.VariableNames))
    error('O arquivo selecionado não contém todas as colunas de entrada necessárias.');
end

% --- Extrair e filtrar sinais de entrada ---
acc_x = filtfilt(b, a, dados.P6_RS_acc_x);
acc_y = filtfilt(b, a, dados.P6_RS_acc_y);
acc_z = filtfilt(b, a, dados.P6_RS_acc_z);

% --- Normalização Z-score ---
acc_x = (acc_x - mean(acc_x)) / std(acc_x);
acc_y = (acc_y - mean(acc_y)) / std(acc_y);
acc_z = (acc_z - mean(acc_z)) / std(acc_z);

% --- Criar janelas temporais ---
time_window = 30;
X_pred = {};
for i = 1:(length(acc_x) - time_window)
    janela = [acc_x(i:i + time_window - 1), ...
              acc_y(i:i + time_window - 1), ...
              acc_z(i:i + time_window - 1)];
    X_pred{end+1} = janela';
end

if isempty(X_pred)
    error('O arquivo não contém dados suficientes para gerar janelas temporais.');
end

% --- Verificar e carregar GRF real ---
tem_dados_reais = ismember(output_column, dados.Properties.VariableNames);
if tem_dados_reais
    grf_real = filtfilt(b, a, dados.rightTotalForce_N_);
    grf_real = grf_real(time_window:end);
end

% --- Figura com subplots verticais ---
figure('Units', 'normalized', 'Position', [0.1 0.05 0.6 0.85]);

for i = 1:length(models)
    % --- Predição ---
    load(models{i}, 'net');
    Y_pred = predict(net, X_pred);

    % --- Dessnormalização das predições ---
    Y_pred = (Y_pred .* std_grf) + mean_grf;

    % --- Ajustar comprimentos para comparação com GRF real ---
    if tem_dados_reais
        min_len = min(length(Y_pred), length(grf_real));
        Y_pred = Y_pred(1:min_len);
        grf_real_trimmed = grf_real(1:min_len);
        predicoes{i} = Y_pred;

        % --- Métricas ---
        rmse = sqrt(mean((grf_real_trimmed - Y_pred).^2));
        rRMSE = (rmse / (max(grf_real_trimmed) - min(grf_real_trimmed))) * 100;
        ss_total = sum((grf_real_trimmed - mean(grf_real_trimmed)).^2);
        ss_residual = sum((grf_real_trimmed - Y_pred).^2);
        r2 = 1 - (ss_residual / ss_total);
        metricas(i, :) = [rmse, rRMSE, r2];

        % --- Subplot ---
        subplot(3, 1, i);
        plot(grf_real_trimmed, 'k', 'LineWidth', 2); hold on;
        plot(Y_pred, 'r', 'LineWidth', 1.8);
        title([model_names{i}, ' vs GRF Real'], 'FontWeight', 'bold');
        xlabel('Samples');
        ylabel('GRF (N)');
        grid on;
    else
        warning('GRF real não encontrado. Métricas não serão calculadas.');
        predicoes{i} = Y_pred;
    end
end

% --- Salvar figura ---
output_fig = fullfile(path, 'subplots_verticais_preto_vermelho_sem_legenda_metrica.png');
exportgraphics(gcf, output_fig, 'Resolution', 600);
disp(['Figura com subplots verticais salva em: ', output_fig]);

% --- Salvar métricas ---
if tem_dados_reais
    metricas_tabela = array2table(metricas, 'VariableNames', {'RMSE', 'rRMSE', 'R2'});
    metricas_tabela.Arquitetura = model_names';
    output_metricas = fullfile(path, 'metricas_comparacao_modelos.csv');
    writetable(metricas_tabela, output_metricas);
    disp('--- Métricas de Comparação ---');
    disp(metricas_tabela);
    disp(['Métricas salvas no arquivo: ', output_metricas]);
end

% --- Salvar predições ---
if tem_dados_reais
    resultado_tabela = table((1:length(predicoes{1}))', predicoes{1}, predicoes{2}, predicoes{3}, grf_real_trimmed(1:length(predicoes{1})), ...
        'VariableNames', {'Amostra', 'Pred_BiLSTM', 'Pred_TCN', 'Pred_Hibrida', 'GRF_Real'});
else
    resultado_tabela = table((1:length(predicoes{1}))', predicoes{1}, predicoes{2}, predicoes{3}, ...
        'VariableNames', {'Amostra', 'Pred_BiLSTM', 'Pred_TCN', 'Pred_Hibrida'});
end

output_predicoes = fullfile(path, 'predicoes_comparacao.csv');
writetable(resultado_tabela, output_predicoes);
disp(['Predições salvas no arquivo: ', output_predicoes]);

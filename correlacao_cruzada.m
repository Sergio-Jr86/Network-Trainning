clear; clc; close all;

% --- Selecionar a pasta com os arquivos CSV ---
folder_path = uigetdir('', 'Selecione a pasta com os arquivos CSV');
if folder_path == 0
    error('Nenhuma pasta selecionada. O script será encerrado.');
end

% --- Listar arquivos CSV na pasta ---
file_list = dir(fullfile(folder_path, '*.csv'));
arquivos = {file_list.name};

% --- Definir Sensores e GRF ---
sensores = {'Lower Back', 'Thigh', 'Leg', 'Foot'};
colunas_acelerometro = {
    {'P6_SA_acc_x', 'P6_SA_acc_y', 'P6_SA_acc_z'}, % Lombar
    {'P6_RT_acc_x', 'P6_RT_acc_y', 'P6_RT_acc_z'}, % Coxa
    {'P6_RS_acc_x', 'P6_RS_acc_y', 'P6_RS_acc_z'}, % Perna
    {'P6_RF_acc_x', 'P6_RF_acc_y', 'P6_RF_acc_z'}  % Pé
};
coluna_grf = 'leftTotalForce_N_'; % Força de reação do solo do pé esquerdo

% --- Parâmetros do Filtro Passa-Baixa ---
fs = 100;  % Frequência de amostragem (Hz)
fc = 10;   % Frequência de corte (Hz)
[b, a] = butter(4, fc / (fs / 2), 'low'); % Filtro de Butterworth de 4ª ordem

% --- Inicializar Vetores para Armazenamento das Correlações ---
correlacoes_total = zeros(length(sensores), length(arquivos));

% --- Processar cada arquivo na pasta ---
for j = 1:length(arquivos)
    arquivo_csv = fullfile(folder_path, arquivos{j});
    disp(['Processando arquivo: ', arquivo_csv]);
    
    % --- Ler os dados do CSV ---
    dados = readtable(arquivo_csv);
    
    % --- Filtrar GRF ---
    if ismember(coluna_grf, dados.Properties.VariableNames)
        grf = filtfilt(b, a, dados.(coluna_grf));
    else
        warning(['Coluna ', coluna_grf, ' não encontrada no arquivo ', arquivos{j}, '. Pulando este arquivo.']);
        continue;
    end
    
    % --- Calcular Correlação Cruzada para Cada Sensor ---
    for i = 1:length(sensores)
        % Verificar se as colunas existem no arquivo
        if all(ismember(colunas_acelerometro{i}, dados.Properties.VariableNames))
            % Obter e filtrar os dados do acelerômetro
            acc_x = filtfilt(b, a, dados.(colunas_acelerometro{i}{1}));
            acc_y = filtfilt(b, a, dados.(colunas_acelerometro{i}{2}));
            acc_z = filtfilt(b, a, dados.(colunas_acelerometro{i}{3}));
            
            % Calcular aceleração resultante
            acc_resultante = sqrt(acc_x.^2 + acc_y.^2 + acc_z.^2);
            
            % Calcular correlação cruzada
            [c, ~] = xcorr(acc_resultante, grf, 'normalized');
            correlacoes_total(i, j) = max(abs(c)); % Armazena o valor máximo de correlação
        else
            warning(['Colunas do sensor ', sensores{i}, ' não encontradas no arquivo ', arquivos{j}, '.']);
        end
    end
end

% --- Calcular Média e Desvio Padrão das Correlações ---
correlacoes_medias = mean(correlacoes_total, 2, 'omitnan');
correlacoes_desvio = std(correlacoes_total, 0, 2, 'omitnan');

% --- Identificar o Sensor com Maior Correlação Média ---
[~, melhor_sensor_idx] = max(correlacoes_medias);
melhor_sensor = sensores{melhor_sensor_idx};

% --- Criar Tabela e Salvar os Resultados ---
resultados_tabela = table(sensores', correlacoes_medias, correlacoes_desvio, ...
    'VariableNames', {'Sensor', 'Correlacao_Media', 'Desvio_Padrao'});

output_filename = fullfile(folder_path, 'Resultados_Correlacao_Cruzada.csv');
writetable(resultados_tabela, output_filename);

% --- Exibir Resultados ---
disp('--- Resultados da Correlação Cruzada ---');
disp(resultados_tabela);
disp(['O sensor com maior correlação média com a GRF é: ', melhor_sensor]);

% --- Plotar os resultados ---
figure;
bar(correlacoes_medias);
hold on;
errorbar(1:length(sensores), correlacoes_medias, correlacoes_desvio, 'k', 'LineStyle', 'none', 'LineWidth', 1.5);
hold off;
set(gca, 'XTickLabel', sensores, 'FontSize', 12);
xlabel('Inertial Sensor', 'FontSize', 14);
ylabel('Average Correlation with GRF', 'FontSize', 14);
title('Cross-Correlation between Inertial Sensors and GRF', 'FontSize', 16);
grid on;

% --- Salvar a figura ---
saveas(gcf, fullfile(folder_path, 'Correlacao_Cruzada.png'));

disp('Gráfico de correlação cruzada salvo com sucesso.');

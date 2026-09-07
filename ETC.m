%% ETC_SMAP_ASCAT_GLDAS.m
% Extended Triple Collocation (ETC) analysis for the triplet
% ASCAT + SMAP + GLDAS.
%
% This script estimates pixel-wise ETC random error variance, RMSE, and
% correlation with the unknown truth for daily soil moisture during
% 2015-2024.
%
% Input assumptions
% -----------------
% 1. All products have been preprocessed to the same 36-km EASE-Grid 2.0.
% 2. Each yearly MAT file has size [lon x lat x 366].
% 3. For non-leap years, layer 60 is a NaN placeholder for 29 February.
% 4. Daily anomalies are calculated using a centered 30-day window
%    [t-14, t+15].
%
% Expected files
% --------------
% ASCAT/ASCATsm_yearYYYY.mat    variable: ASCATsm
% SMAP/sm_yea_DYYYY.mat        variable: sm_yea_D
% GLDAS/GLDASsm_yearYYYY.mat   variable: GLDASsm
% Lat_SMAP_36.mat
% Lon_SMAP_36.mat
%
% Output
% ------
% ETC_ASCAT_SMAP_GLDAS.mat
%
% MATLAB: R2024a or later

%% Configuration

years = 2015:2024;

% Edit this path before running.
data_root = 'PATH_TO_PREPROCESSED_INPUT_DATA';

dir_ascat = fullfile(data_root, 'ASCAT');
dir_smap  = fullfile(data_root, 'SMAP');
dir_gldas = fullfile(data_root, 'GLDAS');

lat_file = fullfile(data_root, 'Lat_SMAP_36.mat');
lon_file = fullfile(data_root, 'Lon_SMAP_36.mat');

output_dir = fullfile(pwd, 'output');
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end
out_file = fullfile(output_dir, 'ETC_ASCAT_SMAP_GLDAS.mat');

% ETC settings
min_triplets = 100;
min_half_obs = 3;
win_back     = 14;
win_forward  = 15;

% Time-axis convention
days_per_year = 366;

% Longitude block size used to limit memory usage
block_size_lon = 50;

%% Grid information

Lat = load_single_variable(lat_file);
Lon = load_single_variable(lon_file);

[nlon, nlat] = size(Lat);
nt_total = numel(years) * days_per_year;

%% Output arrays

RMSE_ASCAT = nan(nlon, nlat, 'single');
RMSE_SMAP  = nan(nlon, nlat, 'single');
RMSE_GLDAS = nan(nlon, nlat, 'single');

R_ASCAT = nan(nlon, nlat, 'single');
R_SMAP  = nan(nlon, nlat, 'single');
R_GLDAS = nan(nlon, nlat, 'single');

ERRVAR_ASCAT = nan(nlon, nlat, 'single');
ERRVAR_SMAP  = nan(nlon, nlat, 'single');
ERRVAR_GLDAS = nan(nlon, nlat, 'single');

N_valid = zeros(nlon, nlat, 'uint16');

%% ETC analysis

n_blocks = ceil(nlon / block_size_lon);

for ib = 1:n_blocks

    ix1 = (ib - 1) * block_size_lon + 1;
    ix2 = min(ib * block_size_lon, nlon);
    ix  = ix1:ix2;

    fprintf('ETC block %d/%d: longitude index %d-%d\n', ...
        ib, n_blocks, ix1, ix2);

    % Read the current longitude block for all years.
    ASCAT_blk = read_multiyear_block( ...
        dir_ascat, years, 'ASCATsm', 'ASCATsm_year%d.mat', ...
        ix, nlat, days_per_year);

    SMAP_blk = read_multiyear_block( ...
        dir_smap, years, 'sm_yea_D', 'sm_yea_D%d.mat', ...
        ix, nlat, days_per_year);

    GLDAS_blk = read_multiyear_block( ...
        dir_gldas, years, 'GLDASsm', 'GLDASsm_year%d.mat', ...
        ix, nlat, days_per_year);

    % Remove the local seasonal background using a centered 30-day window.
    ASCAT_anom = moving_window_anomaly( ...
        ASCAT_blk, win_back, win_forward, min_half_obs);
    SMAP_anom = moving_window_anomaly( ...
        SMAP_blk, win_back, win_forward, min_half_obs);
    GLDAS_anom = moving_window_anomaly( ...
        GLDAS_blk, win_back, win_forward, min_half_obs);

    clear ASCAT_blk SMAP_blk GLDAS_blk

    % Pixel-wise ETC.
    for ii = 1:numel(ix)

        i_global = ix(ii);

        for jj = 1:nlat

            A = squeeze(ASCAT_anom(ii, jj, :));
            S = squeeze(SMAP_anom(ii, jj, :));
            G = squeeze(GLDAS_anom(ii, jj, :));

            valid = isfinite(A) & isfinite(S) & isfinite(G);
            nv = sum(valid);
            N_valid(i_global, jj) = uint16(nv);

            if nv < min_triplets
                continue
            end

            A = double(A(valid));
            S = double(S(valid));
            G = double(G(valid));

            % Positive pairwise association is required by the ETC
            % formulation used here.
            Rpair = corrcoef([A, S, G]);
            if ~(Rpair(1,2) > 0 && Rpair(1,3) > 0 && Rpair(2,3) > 0)
                continue
            end

            C = cov([A, S, G]);

            var_A = C(1,1);
            var_S = C(2,2);
            var_G = C(3,3);

            cov_AS = C(1,2);
            cov_AG = C(1,3);
            cov_SG = C(2,3);

            if ~(var_A > 0 && var_S > 0 && var_G > 0 && ...
                 cov_AS > 0 && cov_AG > 0 && cov_SG > 0)
                continue
            end

            % ETC random error variance.
            err_A = var_A - (cov_AS * cov_AG) / cov_SG;
            err_S = var_S - (cov_AS * cov_SG) / cov_AG;
            err_G = var_G - (cov_AG * cov_SG) / cov_AS;

            if ~(err_A > 0 && err_S > 0 && err_G > 0)
                continue
            end

            % ETC correlation with the unknown truth.
            rho2_A = (cov_AS * cov_AG) / (var_A * cov_SG);
            rho2_S = (cov_AS * cov_SG) / (var_S * cov_AG);
            rho2_G = (cov_AG * cov_SG) / (var_G * cov_AS);

            if ~(rho2_A > 0 && rho2_S > 0 && rho2_G > 0)
                continue
            end

            rho_A = sqrt(rho2_A);
            rho_S = sqrt(rho2_S);
            rho_G = sqrt(rho2_G);

            if ~(rho_A <= 1 && rho_S <= 1 && rho_G <= 1)
                continue
            end

            ERRVAR_ASCAT(i_global, jj) = single(err_A);
            ERRVAR_SMAP(i_global, jj)  = single(err_S);
            ERRVAR_GLDAS(i_global, jj) = single(err_G);

            RMSE_ASCAT(i_global, jj) = single(sqrt(err_A));
            RMSE_SMAP(i_global, jj)  = single(sqrt(err_S));
            RMSE_GLDAS(i_global, jj) = single(sqrt(err_G));

            R_ASCAT(i_global, jj) = single(rho_A);
            R_SMAP(i_global, jj)  = single(rho_S);
            R_GLDAS(i_global, jj) = single(rho_G);

        end
    end

    clear ASCAT_anom SMAP_anom GLDAS_anom
end

%% Save

save(out_file, ...
    'RMSE_ASCAT', 'RMSE_SMAP', 'RMSE_GLDAS', ...
    'R_ASCAT', 'R_SMAP', 'R_GLDAS', ...
    'ERRVAR_ASCAT', 'ERRVAR_SMAP', 'ERRVAR_GLDAS', ...
    'N_valid', 'Lat', 'Lon', 'years', ...
    'min_triplets', 'min_half_obs', 'win_back', 'win_forward', ...
    'days_per_year', '-v7.3');

fprintf('Saved ETC results: %s\n', out_file);

%% Local functions

function X_anom = moving_window_anomaly(X, win_back, win_forward, min_half_obs)
%MOVING_WINDOW_ANOMALY Calculate centered moving-window anomalies.
%
% X'(t) = X(t) - mean[X(t-win_back : t+win_forward)]
%
% An anomaly is retained only when:
%   1. X(t) is valid;
%   2. at least min_half_obs valid observations occur before t; and
%   3. at least min_half_obs valid observations occur after t.

    X = single(X);
    valid = isfinite(X);

    X0 = X;
    X0(~valid) = 0;

    win_sum = movsum(X0, [win_back, win_forward], 3, ...
        'Endpoints', 'shrink');
    win_count = movsum(single(valid), [win_back, win_forward], 3, ...
        'Endpoints', 'shrink');

    win_mean = win_sum ./ win_count;
    win_mean(win_count == 0) = NaN;

    before_count = movsum(single(valid), [win_back, 0], 3, ...
        'Endpoints', 'shrink') - single(valid);

    after_count = movsum(single(valid), [0, win_forward], 3, ...
        'Endpoints', 'shrink') - single(valid);

    keep = valid & ...
        before_count >= min_half_obs & ...
        after_count >= min_half_obs;

    X_anom = X - win_mean;
    X_anom(~keep) = NaN;
end

function X_all = read_multiyear_block( ...
    data_dir, years, varname, file_pattern, ix, nlat, days_per_year)
%READ_MULTIYEAR_BLOCK Read one longitude block from yearly MAT files.

    n_years = numel(years);
    X_all = nan(numel(ix), nlat, n_years * days_per_year, 'single');

    for k = 1:n_years

        yy = years(k);
        file = fullfile(data_dir, sprintf(file_pattern, yy));

        S = load(file, varname);
        X = S.(varname);

        t1 = (k - 1) * days_per_year + 1;
        t2 = k * days_per_year;

        X_all(:, :, t1:t2) = single(X(ix, 1:nlat, 1:days_per_year));
    end
end

function X = load_single_variable(file)
%LOAD_SINGLE_VARIABLE Load the only variable stored in a MAT file.

    S = load(file);
    names = fieldnames(S);
    X = S.(names{1});
end

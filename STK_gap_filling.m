%% STK_gap_filling.m
% Spatio-temporal kriging (STK) gap filling for the fused SMAP-NNsm
% soil moisture product.
%
% Core workflow
% -------------
% 1. Apply the freeze-thaw constraint to define valid observations and
%    missing thawed-soil targets.
% 2. Estimate a multi-year seasonal climatology and calculate residuals.
% 3. Estimate local seasonal space-time covariance parameters from
%    empirical variograms.
% 4. Fill remaining thawed-soil gaps using local ordinary STK.
%
% Notes
% -----
% - Existing valid observations are retained.
% - Only missing pixels satisfying the thawed-soil condition are filled.
% - Filled values are not reused as kriging neighbors.
% - Implementation-specific tuning values are intentionally omitted from
%   this public core version and should be supplied by the user.
%
% MATLAB R2024a or later

clear; clc;

%% Configuration
% Fill in the required paths and parameter values before running.

config = struct();

% Paths and file structure
config.fusion_dir = 'PATH_TO_FUSED_SOIL_MOISTURE';
config.ft_dir = 'PATH_TO_FREEZE_THAW_DATA';
config.output_dir = 'PATH_TO_OUTPUT';

config.lon_file = 'PATH_TO_LONGITUDE_MAT';
config.lat_file = 'PATH_TO_LATITUDE_MAT';

config.lon_variable = 'Lon_SMAP_36';
config.lat_variable = 'Lat_SMAP_36';

config.fusion_file_pattern = 'fusion_%d.mat';
config.ft_file_pattern = 'FT_year%d.mat';
config.output_file_pattern = 'STK_global_year%d.mat';

config.fusion_variable = 'fusion_SM';
config.ft_variable = 'FT_year';

% Time axis
config.years = [];
config.days_per_year = [];
config.leap_placeholder_day = [];

% Freeze-thaw constraint
config.thawed_code = [];

% Global tiling
config.tile_core_nlon = [];
config.tile_core_nlat = [];
config.buffer_n = [];

% Seasonal climatology
config.trend_half_window = [];
config.trend_stat = 'median';

% Parameter estimation
config.min_param_sample = [];

config.variogram_space_lags = [];
config.variogram_time_lags = [];

config.max_time_slices = [];
config.max_pairs_per_bin = [];
config.min_pairs_per_bin = [];
config.min_bins_for_fit = [];

config.lag_bin_tolerance = [];

config.range_s_initial = [];
config.range_t_initial = [];

config.range_s_bounds = [];
config.range_t_bounds = [];

config.psill_initial_factor = [];
config.psill_factor_bounds = [];

config.default_nugget_fraction = [];
config.nugget_fraction_bounds = [];

config.default_space_fraction = [];
config.default_time_fraction = [];
config.space_fraction_bounds = [];

config.k_initial_factor = [];
config.k_factor_bounds = [];

% Local STK neighborhood
config.search_radius_space = [];
config.search_radius_time = [];

config.min_neighbors = [];
config.max_neighbors = [];

config.time_to_space_scale = [];
config.kriging_regularization = [];

% Output range
config.sm_lower_bound = [];
config.sm_upper_bound = [];

%% Grid and time information

Lon = load_named_variable(config.lon_file, config.lon_variable);
Lat = load_named_variable(config.lat_file, config.lat_variable);

[nLon, nLat] = size(Lon);
years = config.years(:)';
nYear = numel(years);
nDay = config.days_per_year;

dummy_day = false(nDay, nYear);
for iy = 1:nYear
    if ~is_leap_year(years(iy))
        dummy_day(config.leap_placeholder_day, iy) = true;
    end
end

config.nLon = nLon;
config.nLat = nLat;
config.nYear = nYear;
config.nDay = nDay;
config.dummy_day = dummy_day;
config.season_id_by_day = fixed_day_to_season(nDay);

%% Output and tiles

if ~exist(config.output_dir, 'dir')
    mkdir(config.output_dir);
end

tile_list = build_tiles( ...
    nLon, nLat, ...
    config.tile_core_nlon, ...
    config.tile_core_nlat, ...
    config.buffer_n);

out_files = cell(nYear, 1);
for iy = 1:nYear
    out_files{iy} = fullfile( ...
        config.output_dir, ...
        sprintf(config.output_file_pattern, years(iy)));

    if exist(out_files{iy}, 'file')
        delete(out_files{iy});
    end

    prepare_year_output(out_files{iy}, nLon, nLat, nDay);
end

%% Global tile processing

for itile = 1:numel(tile_list)

    tile = tile_list(itile);

    fprintf('STK tile %d/%d\n', itile, numel(tile_list));

    [SM_all, FT_all] = load_tile_all_years( ...
        config, years, tile.lon_buf_idx, tile.lat_buf_idx);

    obs_all = isfinite(SM_all) & (FT_all == config.thawed_code);

    for iy = 1:nYear
        if dummy_day(config.leap_placeholder_day, iy)
            obs_all(:, :, config.leap_placeholder_day, iy) = false;
        end
    end

    % Multi-year seasonal climatology.
    SM_clim_input = SM_all;
    SM_clim_input(~obs_all) = NaN;

    C_clim = compute_climatology( ...
        SM_clim_input, ...
        config.trend_half_window, ...
        config.trend_stat);

    % Residual field.
    RES_all = nan(size(SM_all), 'single');
    C4 = repmat(C_clim, 1, 1, 1, nYear);
    RES_all(obs_all) = SM_all(obs_all) - C4(obs_all);

    clear C4 SM_clim_input

    % Local seasonal product-sum covariance parameters.
    params_by_season = estimate_product_sum_params( ...
        RES_all, obs_all, config);

    % Fill each year and write only the tile core.
    for iy = 1:nYear

        SM_filled_core = run_stk_for_year_tile( ...
            iy, SM_all, FT_all, RES_all, C_clim, ...
            obs_all, params_by_season, config, tile);

        M = matfile(out_files{iy}, 'Writable', true);
        M.fusion_SM_STK( ...
            tile.lon_core_idx, ...
            tile.lat_core_idx, :) = SM_filled_core;

        clear M SM_filled_core
    end

    clear SM_all FT_all obs_all C_clim RES_all params_by_season
end

fprintf('STK gap filling completed.\n');


%% ========================================================================
% Local functions
% ========================================================================

function tile_list = build_tiles(nLon, nLat, core_nlon, core_nlat, buffer_n)

lon_starts = 1:core_nlon:nLon;
lat_starts = 1:core_nlat:nLat;

tile_list = struct([]);
it = 0;

for ilon = 1:numel(lon_starts)

    lon1 = lon_starts(ilon);
    lon2 = min(nLon, lon1 + core_nlon - 1);

    for ilat = 1:numel(lat_starts)

        lat1 = lat_starts(ilat);
        lat2 = min(nLat, lat1 + core_nlat - 1);

        lon_core = lon1:lon2;
        lat_core = lat1:lat2;

        lon_buf = max(1, lon1-buffer_n):min(nLon, lon2+buffer_n);
        lat_buf = max(1, lat1-buffer_n):min(nLat, lat2+buffer_n);

        it = it + 1;

        tile_list(it).lon_core_idx = lon_core;
        tile_list(it).lat_core_idx = lat_core;
        tile_list(it).lon_buf_idx = lon_buf;
        tile_list(it).lat_buf_idx = lat_buf;
        tile_list(it).core_lon_in_buf = (lon_core - lon_buf(1)) + 1;
        tile_list(it).core_lat_in_buf = (lat_core - lat_buf(1)) + 1;
    end
end
end


function prepare_year_output(out_file, nLon, nLat, nDay)

M = matfile(out_file, 'Writable', true);
M.fusion_SM_STK(nLon, nLat, nDay) = single(NaN);

end


function [SM_all, FT_all] = load_tile_all_years(config, years, lon_idx, lat_idx)

nYear = numel(years);
nDay = config.days_per_year;

nx = numel(lon_idx);
ny = numel(lat_idx);

SM_all = nan(nx, ny, nDay, nYear, 'single');
FT_all = zeros(nx, ny, nDay, nYear, 'uint8');

for iy = 1:nYear

    yy = years(iy);

    fusion_file = fullfile( ...
        config.fusion_dir, sprintf(config.fusion_file_pattern, yy));

    ft_file = fullfile( ...
        config.ft_dir, sprintf(config.ft_file_pattern, yy));

    SM_all(:, :, :, iy) = single(load_subset( ...
        fusion_file, config.fusion_variable, ...
        lon_idx, lat_idx, 1:nDay));

    FT_all(:, :, :, iy) = uint8(load_subset( ...
        ft_file, config.ft_variable, ...
        lon_idx, lat_idx, 1:nDay));
end
end


function sub = load_subset(mat_file, var_name, idx1, idx2, idx3)

try
    M = matfile(mat_file);
    sub = M.(var_name)(idx1, idx2, idx3);
catch
    S = load(mat_file, var_name);
    X = S.(var_name);
    sub = X(idx1, idx2, idx3);
end
end


function X = load_named_variable(file, var_name)

S = load(file, var_name);
X = S.(var_name);

end


function tf = is_leap_year(yy)

tf = (mod(yy,4) == 0 && mod(yy,100) ~= 0) || mod(yy,400) == 0;

end


function season_id = fixed_day_to_season(nDay)

d0 = datetime(2020,1,1);
month_by_day = zeros(nDay,1);

for d = 1:nDay
    month_by_day(d) = month(d0 + days(d-1));
end

season_id = zeros(nDay,1);
season_id(ismember(month_by_day, [3 4 5]))   = 1;
season_id(ismember(month_by_day, [6 7 8]))   = 2;
season_id(ismember(month_by_day, [9 10 11])) = 3;
season_id(ismember(month_by_day, [12 1 2]))  = 4;

end


function win = circular_day_window(d, half_window, nDay)

raw = (d-half_window):(d+half_window);
win = mod(raw-1, nDay) + 1;

end


function C_clim = compute_climatology(SM, half_window, method_name)

[nx, ny, nDay, nYear] = size(SM);
nPix = nx * ny;

SM2 = reshape(SM, nPix, nDay, nYear);
C2 = nan(nPix, nDay, 'single');

for d = 1:nDay

    win = circular_day_window(d, half_window, nDay);
    vals = reshape(SM2(:, win, :), nPix, []);

    switch lower(method_name)
        case 'median'
            C2(:, d) = single(median(vals, 2, 'omitnan'));
        case 'mean'
            C2(:, d) = single(mean(vals, 2, 'omitnan'));
        otherwise
            error('Unsupported climatology statistic.');
    end
end

C_clim = reshape(C2, nx, ny, nDay);

end


function params_by_season = estimate_product_sum_params(RES_all, obs_all, config)

nSeason = max(config.season_id_by_day);
params_by_season = repmat(empty_parameter_struct(), nSeason, 1);

all_vals = double(RES_all(obs_all & isfinite(RES_all)));
annual_variance = var(all_vals, 0, 'omitnan');

for ss = 1:nSeason

    day_sel = find(config.season_id_by_day == ss);

    R = RES_all(:, :, day_sel, :);
    O = obs_all(:, :, day_sel, :);
    vals = double(R(O & isfinite(R)));

    if numel(vals) >= config.min_param_sample
        sill_total = var(vals, 0, 'omitnan');
        day_sel_fit = day_sel;
    else
        sill_total = annual_variance;
        day_sel_fit = 1:config.nDay;
    end

    % Fallback parameters are supplied through config.
    nugget = config.default_nugget_fraction * sill_total;
    remaining = max(sill_total - nugget, eps);

    sill_s = config.default_space_fraction * remaining;
    sill_t = remaining - sill_s;

    range_s = config.range_s_initial;
    range_t = config.range_t_initial;
    k_ps = config.k_initial_factor / max(sill_total, eps);

    % Empirical marginal variograms.
    V_s = estimate_spatial_variogram(RES_all, obs_all, day_sel_fit, config);
    V_t = estimate_temporal_variogram(RES_all, obs_all, day_sel_fit, config);

    [ps_s, r_s, nug_s, ok_s] = fit_exp_variogram( ...
        V_s.lag, V_s.gamma, V_s.count, ...
        sill_total, config, 'space');

    [ps_t, r_t, nug_t, ok_t] = fit_exp_variogram( ...
        V_t.lag, V_t.gamma, V_t.count, ...
        sill_total, config, 'time');

    if ok_s || ok_t

        nug_list = [];
        if ok_s, nug_list(end+1) = nug_s; end %#ok<AGROW>
        if ok_t, nug_list(end+1) = nug_t; end %#ok<AGROW>

        if ~isempty(nug_list)
            nugget = median(nug_list);
        end

        remaining = max(sill_total - nugget, eps);

        if ~ok_s || ~isfinite(ps_s) || ps_s <= 0
            ps_s = config.default_space_fraction * sill_total;
        end

        if ~ok_t || ~isfinite(ps_t) || ps_t <= 0
            ps_t = config.default_time_fraction * sill_total;
        end

        frac_s = ps_s / max(ps_s + ps_t, eps);
        frac_s = min(config.space_fraction_bounds(2), ...
                     max(config.space_fraction_bounds(1), frac_s));

        sill_s = frac_s * remaining;
        sill_t = (1-frac_s) * remaining;

        if ok_s
            range_s = r_s;
        end

        if ok_t
            range_t = r_t;
        end
    end

    % Joint space-time variogram for product-sum interaction.
    V_st = estimate_st_variogram(RES_all, obs_all, day_sel_fit, config);

    [k_fit, ok_k] = fit_product_sum_k( ...
        V_st.h, V_st.u, V_st.gamma, V_st.count, ...
        sill_s, sill_t, range_s, range_t, nugget, ...
        sill_total, config);

    if ok_k
        k_ps = k_fit;
    end

    params_by_season(ss).sill_total = sill_total;
    params_by_season(ss).sill_s = sill_s;
    params_by_season(ss).sill_t = sill_t;
    params_by_season(ss).nugget = nugget;
    params_by_season(ss).range_s_grid = range_s;
    params_by_season(ss).range_t_day = range_t;
    params_by_season(ss).k = k_ps;
end
end


function p = empty_parameter_struct()

p = struct( ...
    'sill_total', NaN, ...
    'sill_s', NaN, ...
    'sill_t', NaN, ...
    'nugget', NaN, ...
    'range_s_grid', NaN, ...
    'range_t_day', NaN, ...
    'k', NaN);

end


function V = estimate_spatial_variogram(RES_all, obs_all, day_sel, config)

lags = config.variogram_space_lags(:)';
sum_sq = zeros(size(lags));
count = zeros(size(lags));

[nx, ny, ~, nYear] = size(RES_all);
slices = select_time_slices(day_sel, nYear, config.max_time_slices);

max_lag = max(lags);
shifts = build_spatial_shifts(lags, max_lag, config.lag_bin_tolerance);

for is = 1:size(shifts,1)

    dx = shifts(is,1);
    dy = shifts(is,2);
    ibin = shifts(is,3);

    [x1, x2] = shifted_indices(nx, dx);
    [y1, y2] = shifted_indices(ny, dy);

    for it = 1:size(slices,1)

        if count(ibin) >= config.max_pairs_per_bin
            break
        end

        d = slices(it,1);
        iy = slices(it,2);

        A = RES_all(x1,y1,d,iy);
        B = RES_all(x2,y2,d,iy);

        M = obs_all(x1,y1,d,iy) & ...
            obs_all(x2,y2,d,iy) & ...
            isfinite(A) & isfinite(B);

        dv = double(A(M) - B(M));
        [dv, n_use] = limit_candidate_pairs( ...
            dv, config.max_pairs_per_bin - count(ibin));

        sum_sq(ibin) = sum_sq(ibin) + sum(dv.^2);
        count(ibin) = count(ibin) + n_use;
    end
end

gamma = nan(size(lags));
ok = count > 0;
gamma(ok) = 0.5 * sum_sq(ok) ./ count(ok);

V = struct('lag', lags, 'gamma', gamma, 'count', count);

end


function V = estimate_temporal_variogram(RES_all, obs_all, day_sel, config)

lags = config.variogram_time_lags(:)';
sum_sq = zeros(size(lags));
count = zeros(size(lags));

[~, ~, nDay, nYear] = size(RES_all);

day_mask = false(1,nDay);
day_mask(day_sel) = true;

for ilag = 1:numel(lags)

    u = lags(ilag);

    for iy = 1:nYear

        valid_days = day_sel(day_sel + u <= nDay);
        valid_days = valid_days(day_mask(valid_days + u));
        valid_days = limit_time_indices(valid_days, config.max_time_slices);

        for d = valid_days

            if count(ilag) >= config.max_pairs_per_bin
                break
            end

            A = RES_all(:,:,d,iy);
            B = RES_all(:,:,d+u,iy);

            M = obs_all(:,:,d,iy) & ...
                obs_all(:,:,d+u,iy) & ...
                isfinite(A) & isfinite(B);

            dv = double(A(M) - B(M));
            [dv, n_use] = limit_candidate_pairs( ...
                dv, config.max_pairs_per_bin - count(ilag));

            sum_sq(ilag) = sum_sq(ilag) + sum(dv.^2);
            count(ilag) = count(ilag) + n_use;
        end
    end
end

gamma = nan(size(lags));
ok = count > 0;
gamma(ok) = 0.5 * sum_sq(ok) ./ count(ok);

V = struct('lag', lags, 'gamma', gamma, 'count', count);

end


function V = estimate_st_variogram(RES_all, obs_all, day_sel, config)

h_lags = config.variogram_space_lags(:)';
u_lags = config.variogram_time_lags(:)';

sum_sq = zeros(numel(h_lags), numel(u_lags));
count = zeros(numel(h_lags), numel(u_lags));

[nx, ny, nDay, nYear] = size(RES_all);

day_mask = false(1,nDay);
day_mask(day_sel) = true;

shifts = build_spatial_shifts( ...
    h_lags, max(h_lags), config.lag_bin_tolerance);

for iu = 1:numel(u_lags)

    u = u_lags(iu);

    for is = 1:size(shifts,1)

        dx = shifts(is,1);
        dy = shifts(is,2);
        ih = shifts(is,3);

        [x1, x2] = shifted_indices(nx, dx);
        [y1, y2] = shifted_indices(ny, dy);

        for iy = 1:nYear

            valid_days = day_sel(day_sel + u <= nDay);
            valid_days = valid_days(day_mask(valid_days + u));
            valid_days = limit_time_indices( ...
                valid_days, config.max_time_slices);

            for d = valid_days

                if count(ih,iu) >= config.max_pairs_per_bin
                    break
                end

                A = RES_all(x1,y1,d,iy);
                B = RES_all(x2,y2,d+u,iy);

                M = obs_all(x1,y1,d,iy) & ...
                    obs_all(x2,y2,d+u,iy) & ...
                    isfinite(A) & isfinite(B);

                dv = double(A(M) - B(M));
                [dv, n_use] = limit_candidate_pairs( ...
                    dv, config.max_pairs_per_bin - count(ih,iu));

                sum_sq(ih,iu) = sum_sq(ih,iu) + sum(dv.^2);
                count(ih,iu) = count(ih,iu) + n_use;
            end
        end
    end
end

gamma = nan(size(count));
ok = count > 0;
gamma(ok) = 0.5 * sum_sq(ok) ./ count(ok);

[H,U] = ndgrid(h_lags, u_lags);

V = struct('h', H, 'u', U, 'gamma', gamma, 'count', count);

end


function shifts = build_spatial_shifts(lags, max_lag, tolerance)

shifts = [];

for dx = -max_lag:max_lag
    for dy = -max_lag:max_lag

        if dx == 0 && dy == 0
            continue
        end

        if dx < 0 || (dx == 0 && dy < 0)
            continue
        end

        h = sqrt(dx^2 + dy^2);
        [dmin, ibin] = min(abs(lags - h));

        if dmin <= tolerance
            shifts = [shifts; dx, dy, ibin]; %#ok<AGROW>
        end
    end
end
end


function [idx1, idx2] = shifted_indices(n, shift)

if shift >= 0
    idx1 = 1:(n-shift);
    idx2 = (1+shift):n;
else
    idx1 = (1-shift):n;
    idx2 = 1:(n+shift);
end
end


function slices = select_time_slices(day_sel, nYear, max_slices)

[D,Y] = ndgrid(day_sel(:), 1:nYear);
pairs = [D(:), Y(:)];

if size(pairs,1) > max_slices
    idx = round(linspace(1, size(pairs,1), max_slices));
    pairs = pairs(idx,:);
end

slices = pairs;

end


function idx = limit_time_indices(idx, max_n)

if numel(idx) > max_n
    pick = round(linspace(1, numel(idx), max_n));
    idx = idx(pick);
end

end


function [dv, n_use] = limit_candidate_pairs(dv, n_remaining)
% Deterministic evenly spaced sampling is used when the candidate-pair
% count exceeds the prescribed limit.

if isempty(dv) || n_remaining <= 0
    dv = [];
    n_use = 0;
    return
end

if numel(dv) > n_remaining
    idx = round(linspace(1, numel(dv), n_remaining));
    dv = dv(idx);
end

n_use = numel(dv);

end


function [psill, range_val, nugget, ok] = fit_exp_variogram( ...
    lag, gamma, count, sill_total, config, mode_name)

valid = isfinite(lag) & isfinite(gamma) & ...
        count >= config.min_pairs_per_bin & gamma >= 0;

if sum(valid) < config.min_bins_for_fit
    psill = NaN;
    range_val = NaN;
    nugget = NaN;
    ok = false;
    return
end

x = lag(valid);
y = gamma(valid);
w = count(valid) ./ max(count(valid));

switch lower(mode_name)
    case 'space'
        range_bounds = config.range_s_bounds;
        range_initial = config.range_s_initial;
    case 'time'
        range_bounds = config.range_t_bounds;
        range_initial = config.range_t_initial;
    otherwise
        error('Unsupported variogram mode.');
end

psill_bounds = sill_total .* config.psill_factor_bounds;
nugget_bounds = sill_total .* config.nugget_fraction_bounds;

p0 = [ ...
    inv_logit_scale(config.psill_initial_factor*sill_total, psill_bounds), ...
    inv_logit_scale(range_initial, range_bounds), ...
    inv_logit_scale(config.default_nugget_fraction*sill_total, nugget_bounds)];

obj = @(p) exp_variogram_objective( ...
    p, x, y, w, psill_bounds, range_bounds, nugget_bounds);

opts = optimset('Display','off');

try
    p = fminsearch(obj, p0, opts);

    psill = logit_scale(p(1), psill_bounds);
    range_val = logit_scale(p(2), range_bounds);
    nugget = logit_scale(p(3), nugget_bounds);

    ok = isfinite(psill) && isfinite(range_val) && ...
         isfinite(nugget) && psill > 0 && range_val > 0 && nugget >= 0;
catch
    psill = NaN;
    range_val = NaN;
    nugget = NaN;
    ok = false;
end
end


function sse = exp_variogram_objective( ...
    p, x, y, w, psill_bounds, range_bounds, nugget_bounds)

psill = logit_scale(p(1), psill_bounds);
range_val = logit_scale(p(2), range_bounds);
nugget = logit_scale(p(3), nugget_bounds);

yhat = nugget + psill .* (1 - exp(-x ./ range_val));
r = yhat - y;

sse = sum(w .* r.^2) ./ max(sum(w), eps);

end


function val = logit_scale(z, bounds)

lo = bounds(1);
hi = bounds(2);
sig = 1 ./ (1 + exp(-z));
val = lo + (hi-lo).*sig;

end


function z = inv_logit_scale(val, bounds)

lo = bounds(1);
hi = bounds(2);

p = (val-lo) ./ max(hi-lo, eps);
p = min(1-eps, max(eps, p));

z = log(p ./ (1-p));

end


function [k_fit, ok] = fit_product_sum_k( ...
    H, U, gamma_emp, count, ...
    sill_s, sill_t, range_s, range_t, nugget, sill_total, config)

valid = isfinite(H) & isfinite(U) & isfinite(gamma_emp) & ...
        count >= config.min_pairs_per_bin & gamma_emp >= 0;

if sum(valid(:)) < config.min_bins_for_fit
    k_fit = NaN;
    ok = false;
    return
end

h = H(valid);
u = U(valid);
g = gamma_emp(valid);

w = count(valid);
w = w ./ max(w);

Cs = sill_s .* exp(-h ./ range_s);
Ct = sill_t .* exp(-u ./ range_t);

A = nugget + (sill_s-Cs) + (sill_t-Ct);
B = sill_s.*sill_t - Cs.*Ct;

denom = sum(w .* B.^2);

if denom <= 0 || ~isfinite(denom)
    k_fit = NaN;
    ok = false;
    return
end

k_fit = sum(w .* B .* (g-A)) ./ denom;

k_bounds = config.k_factor_bounds ./ max(sill_total, eps);
k_fit = min(k_bounds(2), max(k_bounds(1), k_fit));

ok = isfinite(k_fit);

end


function SM_filled_core = run_stk_for_year_tile( ...
    iy, SM_all, FT_all, RES_all, C_clim, ...
    obs_all, params_by_season, config, tile)

[nxB, nyB, nDay, ~] = size(SM_all);

coreL = tile.core_lon_in_buf;
coreT = tile.core_lat_in_buf;

SM_year  = SM_all(:,:,:,iy);
FT_year  = FT_all(:,:,:,iy);
RES_year = RES_all(:,:,:,iy);
OBS_year = obs_all(:,:,:,iy);

SM_core = SM_year(coreL, coreT, :);
FT_core = FT_year(coreL, coreT, :);

SM_filled_core = SM_core;

for dd = 1:nDay

    if config.dummy_day(dd, iy)
        continue
    end

    target_mask = isnan(SM_core(:,:,dd)) & ...
                  (FT_core(:,:,dd) == config.thawed_code);

    [ic_list, jc_list] = find(target_mask);

    if isempty(ic_list)
        continue
    end

    season_id = config.season_id_by_day(dd);
    params = params_by_season(season_id);

    for kk = 1:numel(ic_list)

        ic = ic_list(kk);
        jc = jc_list(kk);

        ib = coreL(ic);
        jb = coreT(jc);

        [sm_hat, ok] = local_stk_pixel( ...
            ib, jb, dd, ...
            RES_year, OBS_year, C_clim, ...
            params, config, nxB, nyB, nDay);

        if ok
            SM_filled_core(ic,jc,dd) = single(sm_hat);
        end
    end
end
end


function [sm_hat, ok] = local_stk_pixel( ...
    ib, jb, dd, RES_year, OBS_year, C_clim, ...
    params, config, nxB, nyB, nDay)

sm_hat = NaN;
ok = false;

C0 = C_clim(ib,jb,dd);

if ~isfinite(C0)
    return
end

Rs = config.search_radius_space;
Rt = config.search_radius_time;

i1 = max(1, ib-Rs);
i2 = min(nxB, ib+Rs);
j1 = max(1, jb-Rs);
j2 = min(nyB, jb+Rs);
d1 = max(1, dd-Rt);
d2 = min(nDay, dd+Rt);

[ii_vec, jj_vec, tt_vec] = ndgrid(i1:i2, j1:j2, d1:d2);

h0 = sqrt((ii_vec-ib).^2 + (jj_vec-jb).^2);
u0 = abs(tt_vec-dd);

obs_sub = OBS_year(i1:i2, j1:j2, d1:d2);
res_sub = RES_year(i1:i2, j1:j2, d1:d2);

cand = obs_sub & isfinite(res_sub) & ...
       (h0 <= Rs) & (u0 <= Rt);

if ~any(cand(:))
    return
end

x = double(ii_vec(cand));
y = double(jj_vec(cand));
t = double(tt_vec(cand));
h = double(h0(cand));
u = double(u0(cand));
val = double(res_sub(cand));

if numel(val) < config.min_neighbors
    return
end

% Rank candidate neighbors using a combined space-time distance.
d_st = sqrt(h.^2 + (config.time_to_space_scale*u).^2);
[~,ord] = sort(d_st, 'ascend');

if numel(ord) > config.max_neighbors
    ord = ord(1:config.max_neighbors);
end

x = x(ord);
y = y(ord);
t = t(ord);
h = h(ord);
u = u(ord);
val = val(ord);

n = numel(val);

if n < config.min_neighbors
    return
end

Hij = sqrt((x(:)-x(:)').^2 + (y(:)-y(:)').^2);
Uij = abs(t(:)-t(:)');

Cmat = product_sum_covariance(Hij, Uij, params);

diag_idx = 1:n+1:n*n;
Cmat(diag_idx) = Cmat(diag_idx) + ...
                  params.nugget + config.kriging_regularization;

% Target-to-neighbor covariance vector.
k0 = product_sum_covariance(h(:), u(:), params);

A = [Cmat, ones(n,1); ones(1,n), 0];
b = [k0; 1];

try
    sol = A \ b;
catch
    sol = pinv(A) * b;
end

lambda = sol(1:n);

if any(~isfinite(lambda))
    return
end

r_hat = sum(lambda .* val(:));
sm0 = double(C0) + r_hat;

sm0 = min(config.sm_upper_bound, ...
          max(config.sm_lower_bound, sm0));

if isfinite(sm0)
    sm_hat = sm0;
    ok = true;
end
end


function C = product_sum_covariance(h, u, params)
% Product-sum space-time covariance model:
%
%   C_ST(h,u) = k*C_s(h)*C_t(u) + C_s(h) + C_t(u)

Cs = params.sill_s .* exp(-h ./ params.range_s_grid);
Ct = params.sill_t .* exp(-u ./ params.range_t_day);

C = params.k .* Cs .* Ct + Cs + Ct;

end

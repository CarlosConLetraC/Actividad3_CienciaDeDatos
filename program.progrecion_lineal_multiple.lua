import("system", "Math", "json", "File")

local f = File.new("data/datos_test.json", "r")
local src = f:read()
local data = json.decode(src)
local regresionM = Math.regresion_lineal(data)
regresionM:normalizar()
regresionM:entrenar(2, 1e-8, 1e+5)

for i, t in pairs(data) do
	local x = {t[1], t[2]}

	system.print("Predicción:", regresionM:predecir_real(x))
	system.print("Real:", t[3])
	io.write("\n")
end

--[[
local data = {
	{1000, 790, 99},
	{1200, 1160, 95},
	{1000, 929, 95},
	{900, 865, 90},
	{1500, 1140, 105},
	{1000, 929, 105},
	{1400, 1109, 90},
	{1500, 1365, 92},
	{1500, 1112, 98},
	{1600, 1150, 99},
	{1100, 980, 99},
	{1300, 990, 101},
	{1000, 1112, 99},
	{1600, 1252, 94},
	{1600, 1326, 97},
	{1600, 1330, 97},
	{1600, 1365, 99},
	{2200, 1280, 104},
	{1600, 1119, 104},
	{2000, 1328, 105},
	{1600, 1584, 94},
	{2000, 1428, 99},
	{2100, 1365, 99},
	{1600, 1415, 99},
	{2000, 1415, 99}
}
local function entrenar(data, num_features, learning_rate, epochs)
	local n = #data
	
	-- Inicializar coeficientes (b0 incluido)
	local b = {}
	for j = 1, num_features + 1, 1 do
		b[j] = 0
	end

	for e = 1, epochs, 1 do
		local grad = {}
		for j = 1, num_features + 1, 1 do
			grad[j] = 0
		end

		for i = 1, n, 1 do
			local y_real = data[i][num_features + 1]

			-- predicción
			local y_pred = b[1] -- b0
			for j = 1, num_features, 1 do
				y_pred = y_pred + b[j+1] * data[i][j]
			end

			local err = y_pred - y_real

			-- gradientes
			grad[1] = grad[1] + err -- b0
			for j = 1, num_features, 1 do
				grad[j+1] = grad[j+1] + err * data[i][j]
			end
		end

		-- actualizar coeficientes
		for j = 1, num_features + 1, 1 do
			b[j] = b[j] - learning_rate * (grad[j] / n)
		end
	end

	return b
end

local function normalizar(data)
    local x1_mean, x2_mean, y_mean = 0, 0, 0
    local n = #data

    for i = 1, n do
        x1_mean = x1_mean + data[i][1]
        x2_mean = x2_mean + data[i][2]
        y_mean  = y_mean + data[i][3]
    end

    x1_mean = x1_mean / n
    x2_mean = x2_mean / n
    y_mean  = y_mean / n

    local norm = {}

    for i = 1, n do
        norm[i] = {
            data[i][1] - x1_mean,
            data[i][2] - x2_mean,
            data[i][3] - y_mean
        }
    end

    return norm, x1_mean, x2_mean, y_mean
end

local function predecir(b, x)
    local y = b[1] -- b0
    for j = 1, #x, 1 do
        y = y + b[j+1] * x[j]
    end
    return y
end

local function predecir_real(b, x, x1_mean, x2_mean, y_mean)
    local x_norm = {
        x[1] - x1_mean,
        x[2] - x2_mean
    }

    local y_norm = predecir(b, x_norm)

    return y_norm + y_mean
end

local norm_data, x1_mean, x2_mean, y_mean = normalizar(data)
local b = entrenar(norm_data, 2, 1e-8, 100000)

for i, t in pairs(data) do
	local x = {t[1], t[2]}

	system.print("Predicción:", predecir_real(b, x, x1_mean, x2_mean, y_mean))
	system.print("Real:", t[3])
	io.write("\n")
end
]]
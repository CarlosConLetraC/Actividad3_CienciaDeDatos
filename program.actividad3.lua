import("system", "json", "Math", "Table", "File")

-- CONFIG
local LIMIT = 50

local function build_url(page)
	return
		"https://site.web.api.espn.com/apis/common/v3/sports/baseball/mlb/statistics/byathlete" ..
		"?region=us&lang=en&contentorigin=espn&isqualified=true" ..
		"&page=" .. page ..
		"&limit=" .. LIMIT ..
		"&category=batting&sort=batting.avg:desc"
end

-- DESCARGA DE DATOS
local function fetch_page(page)
	local src = system.curldownload(build_url(page))
	return json.decode(src)
end

local all = {}
local page = 1

system.print("Descargando base de datos. . .")
while true do
	local data = fetch_page(page)
	if not data or not data.athletes or #data.athletes == 0 then break end

	table.insert(all, data)
	--system.print("Pagina:", page, "athletes:", #data.athletes)

	if #data.athletes < LIMIT then
		break
	end

	page = page + 1
end
system.print("Base de datos descargada, generando y limpiando dataset. . .")

-- LIMPIEZA (hits vs runs)
local dataset = {}

for _, page in pairs(all) do
	for _, item in ipairs(page.athletes) do
		local cats = item.categories

		if cats then
			for _, c in ipairs(cats) do
				if c.name == "batting" then
					local values = c.values

					local hits = values and values[4]
					local runs = values and values[9]

					if hits and runs then
						table.insert(dataset, {hits, runs})
					end
				end
			end
		end
	end
end

system.print("Dataset limpio:", #dataset)

-- SPLIT TRAIN / TEST
local train, test = {}, {}
local split = math.floor(#dataset * 0.8)

for i = 1, #dataset, 1 do table.insert((i <= split and train) or test, dataset[i]) end

-- MODELO
local modelo = Math.regresion_lineal(train)

modelo:normalizar(1)
modelo:entrenar(1, 0.01, 1000)

-- PEARSON
local r = modelo:pearson(1, 2)
system.print("Pearson (hits vs runs):", r)

-- RMSE (Raiz del Error Cuadratico Medio)
local errores = Table.new()

for i = 1, #test, 1 do
	local pred = modelo:predecir_real({test[i][1]})
	local real = test[i][2]
	errores:insert(pred - real)
end
errores:sort()
local stats = Math.medencen(errores)

system.print("Media del error:", stats.media_aritmetica)
system.print("Desviacion estandar:", stats.desviacion_estandar)
system.print("Varianza:", stats.varianza)

-- RMSE (Raiz del Error Cuadratico Medio)
local sum = 0
for i = 1, #errores, 1 do
	sum = sum + errores[i]*errores[i]
end
local rmse = math.sqrt(sum / #errores)
system.print("RMSE:", rmse)

local export = {
	hits = {},
	runs = {},
	predicciones = {},
	errores = {}
}

for i = 1, #dataset, 1 do
	local x = dataset[i][1]
	local y = dataset[i][2]

	local pred = modelo:predecir_real({x})

	table.insert(export.hits, x)
	table.insert(export.runs, y)
	table.insert(export.predicciones, pred)
	table.insert(export.errores, pred - y)
end

export.metrics = {
	dataset_size = #dataset,
	pearson = r,
	rmse = rmse,
	error_media = stats.media_aritmetica,
	error_std = stats.desviacion_estandar,
	error_var = stats.varianza
}

export.ejemplo_prediccion = {
	input_hits = 20,
	output_runs = modelo:predecir_real({20})
}

local dataset_final = File.new("data/actividad3.json", "wr", true)
dataset_final:clear()
dataset_final:write(json.encode(export))

system.print("dataset `data/actividad3.json` guardado. . .")
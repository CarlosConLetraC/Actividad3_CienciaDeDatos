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

local function fetch_page(page)
	local src = system.curldownload(build_url(page))
	return json.decode(src)
end

-- DESCARGA
local all = {}
local page = 1

system.print("Descargando base de datos. . .")

while true do
	local data = fetch_page(page)

	if not data or not data.athletes or #data.athletes == 0 then
		break
	end

	table.insert(all, data)

	if #data.athletes < LIMIT then
		break
	end

	page = page + 1
end

system.print("Base de datos descargada, generando dataset. . .")

-- FEATURE ENGINEERING
local dataset = {}

for _, pg in ipairs(all) do
	for _, item in ipairs(pg.athletes) do
		local cats = item.categories

		if cats then
			for _, c in ipairs(cats) do
				if c.name == "batting" then
					local v = c.values

					local ab   = tonumber(v and v[2])
					local runs = tonumber(v and v[3])
					local hits = tonumber(v and v[4])
					local hr   = tonumber(v and v[8])
					local bb   = tonumber(v and v[11])

					if ab and hits and runs and ab >= 1 then
						local abf = math.max(ab, 1)

						local hr0 = hr or 0
						local bb0 = bb or 0

						local singles = math.max(hits - hr0, 0)

						local avg = hits / abf
						local slg = (singles + 2 * hr0) / abf
						local iso = slg - avg

						local bb_rate = bb0 / abf
						local power_ratio = hr0 / abf

						local contact = avg
						local clutch = avg * bb_rate * (1 + power_ratio)

						local opi = hits + bb0 + (hr0 * 4)

						local denom = abf + bb0
						local rc = denom > 0 and ((hits + bb0) * (singles + 2 * hr0)) / denom or 0

						table.insert(dataset, {
							contact = contact,
							slg = slg,
							iso = iso,
							power_ratio = power_ratio,
							bb_rate = bb_rate,
							clutch = clutch,
							opi = opi,
							rc = rc,
							runs = runs
						})
					end
				end
			end
		end
	end
end

system.print("Dataset limpio:", #dataset)

-- SPLIT TRAIN / TEST
local train, test = {}, {}
local split = math.min(#dataset - 1, math.floor(#dataset * 0.8))

for i = 1, #dataset, 1 do
	table.insert((i <= split and train) or test, dataset[i])
	--	if i <= split then
--		table.insert(train, dataset[i])
--	else
--		table.insert(test, dataset[i])
--	end
end

-- MODELO
local features = {
	"contact",
	"slg",
	"iso",
	"power_ratio",
	"bb_rate",
	"clutch",
	"opi",
	"rc"
}

local modelo = Math.regresion_lineal(train, {
	features = features,
	target = "runs"
})

modelo:normalizar()
modelo:entrenar(0.005, 1500)

-- PREDICCION + ERRORES
local errores = Table.new()

for i = 1, #test, 1 do
	local row = test[i]
	-- system.print(row)
	local pred = modelo:predecir(row)
	errores:insert(pred - row.runs)
end

local stats = Math.medencen(errores)

system.print("Media error:", stats.media_aritmetica)
system.print("Std error:", stats.desviacion_estandar)
system.print("Var error:", stats.varianza)

-- RMSE
local rmse_modelo = modelo:rmse()

local sum = 0
for i = 1, #test do
	local row = test[i]
	-- system.print(row)
	local pred = modelo:predecir(row)
	local err = pred - row.runs
	sum = sum + err * err
end

local rmse_manual = math.sqrt(sum / #test)

system.print("RMSE modelo:", rmse_modelo)
system.print("RMSE manual:", rmse_manual)

-- PEARSON (contact vs runs)
local r_contact_runs = modelo:pearson(1, #features + 1)
system.print("Pearson contact vs runs:", r_contact_runs)

-- EXPORT
local export = {
	contact = {},
	slg = {},
	iso = {},
	power_ratio = {},
	bb_rate = {},
	clutch = {},
	opi = {},
	rc = {},
	runs = {},
	predicciones = {},
	errores = {}
}

for i = 1, #dataset, 1 do
	local row = dataset[i]
	local pred = modelo:predecir(row)

	table.insert(export.contact, row.contact)
	table.insert(export.slg, row.slg)
	table.insert(export.iso, row.iso)
	table.insert(export.power_ratio, row.power_ratio)
	table.insert(export.bb_rate, row.bb_rate)
	table.insert(export.clutch, row.clutch)
	table.insert(export.opi, row.opi)
	table.insert(export.rc, row.rc)
	table.insert(export.runs, row.runs)
	table.insert(export.predicciones, pred)
	table.insert(export.errores, pred - row.runs)
end

export.metrics = {
	dataset_size = #dataset,
	rmse_modelo = rmse_modelo,
	rmse_manual = rmse_manual,
	r_contact_runs = r_contact_runs,
	error_media = stats.media_aritmetica,
	error_std = stats.desviacion_estandar,
	error_var = stats.varianza
}

-- EJEMPLO DE PREDICCIÓN (REALISTA, NO "hits")
export.ejemplo_prediccion = {
	input = {
		contact = 0.25,
		slg = 0.45,
		iso = 0.18,
		power_ratio = 0.05,
		bb_rate = 0.1,
		clutch = 0.02,
		opi = 120,
		rc = 90
	},
	output_runs = modelo:predecir({
		contact = 0.25,
		slg = 0.45,
		iso = 0.18,
		power_ratio = 0.05,
		bb_rate = 0.1,
		clutch = 0.02,
		opi = 120,
		rc = 90
	})
}

-- SAVE
local dataset_final = File.new("data/actividad3.json", "w", true)
dataset_final:clear()
dataset_final:write(json.encode(export))

system.print("dataset `data/actividad3.json` guardado. . .")
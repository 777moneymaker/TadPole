CONFIG_SCHEMA = {
	"$schema": "http://json-schema.org/draft-07/schema#", 
	"$id": "https://example.com/object1670961194.json", 
	"title": "Root", 
	"type": "object",
	"properties": {
		"molecular_weight": {
			"$id": "#root/molecular_weight", 
			"title": "Molecular_weight", 
			"type": "boolean"
		},
		"instability_index": {
			"$id": "#root/instability_index", 
			"title": "Instability_index", 
			"type": "boolean"
		},
		"isoelectric_point": {
			"$id": "#root/isoelectric_point", 
			"title": "Isoelectric_point", 
			"type": "boolean"
		},
		"gravy": {
			"$id": "#root/gravy", 
			"title": "Gravy", 
			"type": "boolean"
		},
		"aromaticity": {
			"$id": "#root/aromaticity", 
			"title": "Aromaticity", 
			"type": "boolean",
		},
		"distance": {
			"$id": "#root/distance",
			"title": "Distance",
			"type": "number",
		},
		"sequential": {
			"$id": "#root/enumerate",
			"title": "Sequential",
			"type": "string",
			"enum": ["none", "enumerate", "collapse", "consecutive", "encode"]
		}
	},
	"required": [
		"molecular_weight",
		"instability_index",
		"isoelectric_point",
		"gravy",
		"aromaticity",
		"sequential"
	],
	"additionalProperties": False
}

DEFAULT_CONFIG =     { 
	"molecular_weight": True, 
    "instability_index": True,
    "isoelectric_point": True, 
    "gravy": True, 
    "aromaticity": True,
	"distance": float("INF"),
    "sequential": "none" # "none", "enumerate", "collapse", "consecutive", "encode"
}
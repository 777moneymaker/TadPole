config_schema = {
	"$schema": "http://json-schema.org/draft-07/schema#", 
	"$id": "https://example.com/object1670961194.json", 
	"title": "Root", 
	"type": "object",
	"properties": {
		"molecular_weight": {
			"$id": "#root/molecular_weight", 
			"title": "Molecular_weight", 
			"type": "boolean",
			"default": True
		},
		"instability_index": {
			"$id": "#root/instability_index", 
			"title": "Instability_index", 
			"type": "boolean",
			"default": True
		},
		"isoelectric_point": {
			"$id": "#root/isoelectric_point", 
			"title": "Isoelectric_point", 
			"type": "boolean",
			"default": True
		},
		"gravy": {
			"$id": "#root/gravy", 
			"title": "Gravy", 
			"type": "boolean",
			"default": True
		},
		"aromaticity": {
			"$id": "#root/aromaticity", 
			"title": "Aromaticity", 
			"type": "boolean",
			"default": True
		}
	},
	"additionalProperties": False
}

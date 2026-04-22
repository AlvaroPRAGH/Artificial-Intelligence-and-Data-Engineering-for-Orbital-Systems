import json
import csv

dataset_file = "data/raw/orbital_observations.csv"
metadata_file = "data/raw/metadata.json"

with open(metadata_file, 'r', encoding='utf-8') as f:
    metadata = json.load(f)

rows = []
with open(dataset_file, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)

    detected_columns = reader.fieldnames

    for row in reader:
        rows.append(row)

print("--- Inspección de Datos ---")
print(f"Dataset: {metadata['dataset_name']}")
print(f"Records loaded: {len(rows)}")
print(f"Columns (dataset): {detected_columns}")
print(f"Columns (metadata): {metadata['columns']}")

expected_columns = metadata["columns"]
actual_columns = detected_columns


if expected_columns == actual_columns:
    print("Column validation: OK")
else:
    print("Column validation: MISMATCH")
    print(f"Expected: {expected_columns}")
    print(f"Actual: {actual_columns}")


expected_records = metadata["num_records"]
actual_records = len(rows)

if expected_records == actual_records:
    print("Record count: OK")
else:
    print("Record count: MISMATCH")
    print(f"Expected: {expected_records}")
    print(f"Actual: {actual_records}")


valid_records = []
invalid_records = []

for row in rows:
    temp = row.get("temperature", "")
    
    if temp == "INVALID" or temp.strip() == "":
        invalid_records.append(row)
    else:
        valid_records.append(row)

print(f"Valid: {len(valid_records)}")
print(f"Invalid: {len(invalid_records)}")


expected_invalid = metadata.get("invalid_records", 0)

if len(invalid_records) == expected_invalid:
    print("Invalid records check: OK")
else:
    print("Invalid records check: MISMATCH")
    print(f"Expected: {expected_invalid}")
    print(f"Actual: {len(invalid_records)}")


#TASK 8

valid_output_file = "data/processed/observations_valid.csv"
invalid_output_file = "data/processed/observations_invalid.csv"


with open(valid_output_file, 'w', encoding='utf-8', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=detected_columns) 
    
    writer.writeheader()         
    writer.writerows(valid_records) 


with open(invalid_output_file, 'w', encoding='utf-8', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=detected_columns)
    
    writer.writeheader()
    writer.writerows(invalid_records)

print("--- Guardado de Archivos ---")
print(f"Registros válidos guardados en: {valid_output_file}")
print(f"Registros inválidos guardados en: {invalid_output_file}")

#TASK 9
model_input_file = "data/processed/model_input.csv"

feature_columns = metadata["feature_columns"]

with open(model_input_file, 'w', encoding='utf-8', newline='') as f:
  
    writer = csv.DictWriter(f, fieldnames=feature_columns, extrasaction='ignore')
    
    writer.writeheader()
    writer.writerows(valid_records)

print("--- Preparación para el Modelo ---")
print(f"Dataset de entrada al modelo (solo features) guardado en: {model_input_file}")


summary_file = "reports/ingestion_summary.txt"

col_validation_result = "OK" if expected_columns == actual_columns else "MISMATCH"
rec_validation_result = "OK" if expected_records == actual_records else "MISMATCH"
    
report_content = f"""Dataset: {metadata['dataset_name']}
Records loaded: {actual_records}
Expected records: {expected_records}

Column validation: {col_validation_result}
Record count validation: {rec_validation_result}

Valid records: {len(valid_records)}
Invalid records: {len(invalid_records)}

Generated files:
- {valid_output_file}
- {invalid_output_file}
- {model_input_file}
"""

with open(summary_file, 'w', encoding='utf-8') as f:
    f.write(report_content)

print("--- Reporte Generado ---")
print(f"Resumen de ingesta guardado en: {summary_file}")
# Helpers for `baml-cli local recordings query --output-format json`, whose
# payload is {columns: [{name, type}, ...], rows: [[...], ...]}.
# Load with: jq -L scripts 'import "btel_jq" as btel; ...'
#
# named_rows maps the positional rows into JSON objects keyed by the SQL
# column/alias name, so assertions use .span_name, .model_id, .step_index,
# ... instead of brittle positional indices like .[9].
def named_rows:
  [.columns[].name] as $names
  | [.rows[] | [range(0; ($names | length)) as $i | {key: $names[$i], value: .[$i]}] | from_entries];

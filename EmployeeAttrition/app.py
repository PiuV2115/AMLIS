Backend Flask app that accepts:
# scale if scaler available
if scaler:
row_scaled = scaler.transform(row)
else:
row_scaled = row.values


pred = model.predict(row_scaled)
return jsonify({'prediction': int(pred[0])})
except Exception as e:
return jsonify({'error': str(e)}), 500




@app.route('/predict-csv', methods=['POST'])
def predict_csv():
if 'file' not in request.files:
return jsonify({'error':'No file part'}), 400
file = request.files['file']
if file.filename == '':
return jsonify({'error':'No selected file'}), 400
try:
df = pd.read_csv(file)
# ensure expected columns
for col in FEATURE_COLUMNS:
if col not in df.columns:
df[col] = 0
df = df[FEATURE_COLUMNS]


# apply encoders if available
if encoders:
for col, enc in encoders.items():
if col in df.columns:
df[col] = enc.transform(df[col].astype(str))


# scale
if scaler:
arr = scaler.transform(df)
else:
arr = df.values


preds = model.predict(arr)
df_out = df.copy()
df_out['Attrition_Prediction'] = preds


# return as downloadable CSV
buf = io.StringIO()
df_out.to_csv(buf, index=False)
buf.seek(0)
return send_file(io.BytesIO(buf.getvalue().encode()), mimetype='text/csv', as_attachment=True, download_name='predicted_results.csv')


except Exception as e:
return jsonify({'error':str(e)}), 500




if __name__ == '__main__':
app.run(host='0.0.0.0', port=5000, debug=True)
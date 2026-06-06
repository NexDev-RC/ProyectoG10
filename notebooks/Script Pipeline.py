from src.pipeline.train_pipeline import TrainPipeline

pipe = TrainPipeline()

pipe.run(tune=False)

print('Features seleccionadas:', pipe.selected_features)

print('Métricas:', pipe.metrics_report.get('final_model'))
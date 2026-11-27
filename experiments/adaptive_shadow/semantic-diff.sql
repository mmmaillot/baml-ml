SELECT escalation.context_metadata['baml_ml.row_index'] AS row_index,
       stage.span_name AS stage,
       stage.context_metadata['baml_ml.shadow_role'] AS role,
       stage.context_metadata['baml_ml.model_id'] AS model_id,
       stage.input_args AS inputs,
       stage.output_value AS output
FROM spans stage
JOIN spans replay ON stage.parent_span_id = replay.span_id
JOIN spans escalation ON replay.parent_span_id = escalation.span_id
WHERE escalation.span_name = 'user.ShadowEscalatedRow'
  AND replay.span_name = 'user.RunSelectiveModel'
  AND stage.span_name IN ('user.ObservedImpute', 'user.ObservedScale',
      'user.ObservedLinear', 'user.ObservedDecisionScores', 'user.ObservedSoftmax',
      'user.ObservedLabels', 'user.ObservedTreeTraversal', 'user.ObservedForestTraversal')
ORDER BY row_index, stage, role

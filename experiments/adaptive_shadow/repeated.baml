function AdaptiveRepeated(model: string) -> IntArrayResult[] {
    let m = LoadLogisticModel(model, observation = ObservationMode.Adaptive);
    let healthy = m.predict([[2.0]], ["x"]);
    let mixed = m.predict([[2.0], [float.nan()], [5.0], [0.0], [1.0]], ["x"]);
    let repeat = m.predict([[0.0]], ["x"]);
    [healthy, mixed, repeat]
}

import java.util.ArrayList;

public class Layer {
    ArrayList<Neuron> neurons;

    public Layer(int nin, int nout) {
        this.neurons = new ArrayList<>();
        for (int i = 0; i < nout; i++) {
            this.neurons.add(new Neuron(nin));
        }
    }

    public ArrayList<Value> forward(ArrayList<Value> x) {
        ArrayList<Value> out = new ArrayList<>();
        for (Neuron neuron : this.neurons) {
            out.add(neuron.forward(x));
        }
        return out;
    }

    public ArrayList<Value> parameters() {
        ArrayList<Value> params = new ArrayList<>();
        for (Neuron neuron : this.neurons) {
            params.addAll(neuron.parameters());
        }
        return params;
    }
}

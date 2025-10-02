import java.util.ArrayList;
import java.util.Random;

public class Neuron {
    ArrayList<Value> weights;
    Value bias;

    Neuron(int nin) {
        Random random = new Random();
        this.weights = new ArrayList<>();
        for (int i = 0; i < nin; i++) {
            this.weights.add(new Value(random.nextGaussian()));
        }
        this.bias = new Value(0.0);
    }

    public Value forward(ArrayList<Value> x) {
        Value act = this.bias;
        for (int i = 0; i < this.weights.size(); i++) {
            act = act.add(this.weights.get(i).multiply(x.get(i)));
        }
        return act.tanh();
    }

    public ArrayList<Value> parameters() {
        ArrayList<Value> params = new ArrayList<>(this.weights);
        params.add(this.bias);
        return params;
    }
}
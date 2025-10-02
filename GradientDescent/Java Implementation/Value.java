import java.util.*;

public class Value {
    double data;
    double gradient;
    Runnable backward;
    // Runnable is a functional interface with a run() method, so we can use lambda expressions for it
    Set<Value> previous;
    String operation;

    // Initializer for leaf nodes
    Value(double data) {
        this(data, new HashSet<>(), "");
    }

    Value(double data, Set<Value> children, String operation) {
        this.data = data;
        this.gradient = 0.0;
        this.previous = children;
        this.operation = operation;
        this.backward = () -> {
        }; // Default to no-op
    }

    // Display
    @Override
    public String toString() {
        return String.format("Value(data=%.4f, gradient=%.4f)", data, gradient);
    }

    // Addition
    public Value add(Value other) {
        Value out = new Value(this.data + other.data, new HashSet<>(Arrays.asList(this, other)), "+");
        out.backward = () -> {
            this.gradient += 1.0 * out.gradient;
            other.gradient += 1.0 * out.gradient;
        };
        return out;
    }

    // Subtraction
    public Value subtract(Value other) {
        return this.add(other.neg());
    }

    public Value neg() {
        return this.multiply(new Value(-1.0));
    }

    // Multiplication
    public Value multiply(Value other) {
        Value out = new Value(this.data * other.data, new HashSet<>(Arrays.asList(this, other)), "*");
        out.backward = () -> {
            this.gradient += other.data * out.gradient;
            other.gradient += this.data * out.gradient;
        };
        return out;
    }

    // Division
    public Value divide(Value other) {
        return this.multiply(other.power(-1.0));
    }

    // Power
    public Value power(double exponent) {
        Value out = new Value(Math.pow(data, exponent), new HashSet<>(Arrays.asList(this)), "^" + exponent);
        out.backward = () -> {
            this.gradient += exponent * Math.pow(data, exponent - 1) * out.gradient;
        };
        return out;
    }

    // tanh activation
    public Value tanh() {
        double t = Math.tanh(this.data);
        Value out = new Value(t, new HashSet<>(Arrays.asList(this)), "tanh");
        out.backward = () -> {
            this.gradient += (1 - t * t) * out.gradient;
        };
        return out;
    }

    // Backpropagation
    public void backward() {
        List<Value> topo = new ArrayList<>();
        Set<Value> visited = new HashSet<>();
        buildTopo(this, visited, topo);
        this.gradient = 1.0; // Seed the gradient
        Collections.reverse(topo);
        for (Value v : topo) {
            v.backward.run();
        }
    }

    public void buildTopo(Value v, Set<Value> visited, List<Value> topo) {
        if (!visited.contains(v)) {
            visited.add(v);
            for (Value child : v.previous) {
                buildTopo(child, visited, topo);
            }
            topo.add(v);
        }
    }
}
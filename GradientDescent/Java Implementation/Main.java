import java.util.*;

public class Main {
    public static void main(String[] args) {
        // Create network: 4 -> [4,4,1]
        MultipleLayers n = new MultipleLayers(4, new int[] { 4, 4, 1 });

        // Training data
        double[][] xs = {
                { 2.0, 3.0, -1.0, 0.5 },
                { 3.0, -1.0, 0.5, 2.0 },
                { 4.0, 0.5, 2.0, 3.0 },
                { 0.5, 2.0, 3.0, -1.0 }
        };
        double[] ys = { 1.0, -1.0, -1.0, 1.0 };

        // Train for 20 epochs
        for (int k = 0; k < 100; k++) {
            List<Value> ypred = new ArrayList<>();

            // Forward pass
            for (int i = 0; i < xs.length; i++) {
                ArrayList<Value> xvals = new ArrayList<>();
                for (double v : xs[i])
                    xvals.add(new Value(v));
                List<Value> out = n.forward(xvals);
                ypred.add(out.get(0)); // output is single neuron -> 1 value
            }

            // Loss = sum((yout - ygt)^2)
            Value loss = new Value(0.0);
            for (int i = 0; i < ys.length; i++) {
                Value ygt = new Value(ys[i]);
                Value diff = ypred.get(i).subtract(ygt);
                loss = loss.add(diff.power(2));
            }

            // Backward pass
            loss.backward();

            // Update params with SGD
            for (Value p : n.parameters()) {
                p.data += -0.007 * p.gradient;
                p.gradient = 0.0;
            }

            System.out.println("Epoch " + (k + 1) + " | Loss = " + loss.data);
        }
    }
}

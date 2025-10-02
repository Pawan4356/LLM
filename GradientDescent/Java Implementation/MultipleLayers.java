import java.util.ArrayList;

public class MultipleLayers {
    ArrayList<Layer> layers;

    public MultipleLayers(int nin, int[] nouts) {
        layers = new ArrayList<>();
        int prevSize = nin;
        for (int nout : nouts) {
            layers.add(new Layer(prevSize, nout));
            prevSize = nout;
        }
    }

    public ArrayList<Value> forward(ArrayList<Value> x) {
        ArrayList<Value> out = x;
        for (Layer layer : layers) {
            out = layer.forward(out);
        }
        return out;
    }

    public ArrayList<Value> parameters() {
        ArrayList<Value> params = new ArrayList<>();
        for (Layer layer : layers) {
            params.addAll(layer.parameters());
        }
        return params;
    }
}

import org.junit.Before;
import org.junit.Test;


import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Set;

import static org.junit.Assert.*;

/**
 * @author Rishi Soni
 * @version 1.0
 */
public class SuperTest {

    private Graph<Integer> directedGraph;
    private Graph<Character> undirectedGraph;
    public static final int TIMEOUT = 200;

    @Before
    public void init() {
        directedGraph = createDirectedGraph();
        undirectedGraph = createUndirectedGraph();
    }

    /**
     * Creates a directed graph.
     *       1 —— 2 —— 3 —— 6              *Unable to draw arrows
     *        \    \  /    /
     *        5 ——— 4 —— 7
     * @return the completed graph
     */
    private Graph<Integer> createDirectedGraph() {
        Set<Vertex<Integer>> vertices = new HashSet<Vertex<Integer>>();
        for (int i = 1; i <= 7; i++) {
            vertices.add(new Vertex<Integer>(i));
        }

        Set<Edge<Integer>> edges = new LinkedHashSet<Edge<Integer>>();
        edges.add(new Edge<>(new Vertex<>(1), new Vertex<>(2), 3));
        edges.add(new Edge<>(new Vertex<>(2), new Vertex<>(3), 1));
        edges.add(new Edge<>(new Vertex<>(2), new Vertex<>(4), 5));
        edges.add(new Edge<>(new Vertex<>(4), new Vertex<>(5), 4));
        edges.add(new Edge<>(new Vertex<>(5), new Vertex<>(1), 2));
        edges.add(new Edge<>(new Vertex<>(4), new Vertex<>(3), 10));
        edges.add(new Edge<>(new Vertex<>(3), new Vertex<>(6), 8));
        edges.add(new Edge<>(new Vertex<>(6), new Vertex<>(7), 12));
        edges.add(new Edge<>(new Vertex<>(7), new Vertex<>(4), 2));

        return new Graph<Integer>(vertices, edges);
    }

    /**
     * Creates an undirected graph.
     *       A —————— G
     *       |      /  \         (H) - Not connected
     *       C —— B    F
     *       \  /  \  /
     *        E ———— D
     * @return the completed graph
     */
    private Graph<Character> createUndirectedGraph() {
        Set<Vertex<Character>> vertices = new HashSet<>();
        for (int i = 65; i <= 72; i++) {
            vertices.add(new Vertex<Character>((char) i));
        }

        Set<Edge<Character>> edges = new LinkedHashSet<>();
        edges.add(new Edge<>(new Vertex<>('A'), new Vertex<>('G'), 18));
        edges.add(new Edge<>(new Vertex<>('G'), new Vertex<>('A'), 18));

        edges.add(new Edge<>(new Vertex<>('A'), new Vertex<>('C'), 2));
        edges.add(new Edge<>(new Vertex<>('C'), new Vertex<>('A'), 2));

        edges.add(new Edge<>(new Vertex<>('C'), new Vertex<>('B'), 4));
        edges.add(new Edge<>(new Vertex<>('B'), new Vertex<>('C'), 4));

        edges.add(new Edge<>(new Vertex<>('C'), new Vertex<>('E'), 5));
        edges.add(new Edge<>(new Vertex<>('E'), new Vertex<>('C'), 5));

        edges.add(new Edge<>(new Vertex<>('E'), new Vertex<>('B'), 1));
        edges.add(new Edge<>(new Vertex<>('B'), new Vertex<>('E'), 1));

        edges.add(new Edge<>(new Vertex<>('B'), new Vertex<>('D'), 2));
        edges.add(new Edge<>(new Vertex<>('D'), new Vertex<>('B'), 2));

        edges.add(new Edge<>(new Vertex<>('E'), new Vertex<>('D'), 7));
        edges.add(new Edge<>(new Vertex<>('D'), new Vertex<>('E'), 7));

        edges.add(new Edge<>(new Vertex<>('F'), new Vertex<>('D'), 3));
        edges.add(new Edge<>(new Vertex<>('D'), new Vertex<>('F'), 3));

        edges.add(new Edge<>(new Vertex<>('B'), new Vertex<>('G'), 1));
        edges.add(new Edge<>(new Vertex<>('G'), new Vertex<>('B'), 1));

        edges.add(new Edge<>(new Vertex<>('F'), new Vertex<>('G'), 6));
        edges.add(new Edge<>(new Vertex<>('G'), new Vertex<>('F'), 6));

        return new Graph<Character>(vertices, edges);
    }

    @Test(timeout = TIMEOUT)
    public void bfsExceptions() {

        assertThrows(IllegalArgumentException.class, () -> {
            GraphAlgorithms.bfs(null, undirectedGraph);
        });

        assertThrows(IllegalArgumentException.class, () -> {
            GraphAlgorithms.bfs(new Vertex<>('A'), null);
        });

        assertThrows(IllegalArgumentException.class, () -> {
            GraphAlgorithms.bfs(new Vertex<>('Z'), undirectedGraph);
        });
    }

    @Test(timeout = TIMEOUT)
    public void dfsExceptions() {

        assertThrows(IllegalArgumentException.class, () -> {
            GraphAlgorithms.dfs(null, undirectedGraph);
        });

        assertThrows(IllegalArgumentException.class, () -> {
            GraphAlgorithms.dfs(new Vertex<>('A'), null);
        });

        assertThrows(IllegalArgumentException.class, () -> {
            GraphAlgorithms.dfs(new Vertex<>('Z'), undirectedGraph);
        });
    }

    @Test(timeout = TIMEOUT)
    public void dijkstrasExceptions() {

        assertThrows(IllegalArgumentException.class, () -> {
            GraphAlgorithms.dijkstras(null, undirectedGraph);
        });

        assertThrows(IllegalArgumentException.class, () -> {
            GraphAlgorithms.dijkstras(new Vertex<>('A'), null);
        });

        assertThrows(IllegalArgumentException.class, () -> {
            GraphAlgorithms.dijkstras(new Vertex<>('Z'), undirectedGraph);
        });
    }

    @Test(timeout = TIMEOUT)
    public void primsExceptions() {

        assertThrows(IllegalArgumentException.class, () -> {
            GraphAlgorithms.prims(null, undirectedGraph);
        });

        assertThrows(IllegalArgumentException.class, () -> {
            GraphAlgorithms.prims(new Vertex<>('A'), null);
        });

        assertThrows(IllegalArgumentException.class, () -> {
            GraphAlgorithms.prims(new Vertex<>('Z'), undirectedGraph);
        });
    }

    @Test(timeout = TIMEOUT)
    public void bfsUndirectedA() {
        List<Vertex<Character>> bfsActual = GraphAlgorithms.bfs(
                new Vertex<>('A'), undirectedGraph);

        List<Vertex<Character>> bfsExpected = new LinkedList<>();
        bfsExpected.add(new Vertex<>('A'));
        bfsExpected.add(new Vertex<>('G'));
        bfsExpected.add(new Vertex<>('C'));
        bfsExpected.add(new Vertex<>('B'));
        bfsExpected.add(new Vertex<>('F'));
        bfsExpected.add(new Vertex<>('E'));
        bfsExpected.add(new Vertex<>('D'));

        assertEquals(bfsExpected, bfsActual);
    }

    @Test(timeout = TIMEOUT)
    public void bfsUndirectedB() {
        List<Vertex<Character>> bfsActual = GraphAlgorithms.bfs(
                new Vertex<>('B'), undirectedGraph);

        List<Vertex<Character>> bfsExpected = new LinkedList<>();
        bfsExpected.add(new Vertex<>('B'));
        bfsExpected.add(new Vertex<>('C'));
        bfsExpected.add(new Vertex<>('E'));
        bfsExpected.add(new Vertex<>('D'));
        bfsExpected.add(new Vertex<>('G'));
        bfsExpected.add(new Vertex<>('A'));
        bfsExpected.add(new Vertex<>('F'));

        assertEquals(bfsExpected, bfsActual);
    }

    @Test(timeout = TIMEOUT)
    public void bfsUndirectedC() {
        List<Vertex<Character>> bfsActual = GraphAlgorithms.bfs(
                new Vertex<>('C'), undirectedGraph);

        List<Vertex<Character>> bfsExpected = new LinkedList<>();
        bfsExpected.add(new Vertex<>('C'));
        bfsExpected.add(new Vertex<>('A'));
        bfsExpected.add(new Vertex<>('B'));
        bfsExpected.add(new Vertex<>('E'));
        bfsExpected.add(new Vertex<>('G'));
        bfsExpected.add(new Vertex<>('D'));
        bfsExpected.add(new Vertex<>('F'));

        assertEquals(bfsExpected, bfsActual);
    }

    @Test(timeout = TIMEOUT)
    public void bfsUndirectedD() {
        List<Vertex<Character>> bfsActual = GraphAlgorithms.bfs(
                new Vertex<>('D'), undirectedGraph);

        List<Vertex<Character>> bfsExpected = new LinkedList<>();
        bfsExpected.add(new Vertex<>('D'));
        bfsExpected.add(new Vertex<>('B'));
        bfsExpected.add(new Vertex<>('E'));
        bfsExpected.add(new Vertex<>('F'));
        bfsExpected.add(new Vertex<>('C'));
        bfsExpected.add(new Vertex<>('G'));
        bfsExpected.add(new Vertex<>('A'));

        assertEquals(bfsExpected, bfsActual);
    }

    @Test(timeout = TIMEOUT)
    public void bfsUndirectedE() {
        List<Vertex<Character>> bfsActual = GraphAlgorithms.bfs(
                new Vertex<>('E'), undirectedGraph);

        List<Vertex<Character>> bfsExpected = new LinkedList<>();
        bfsExpected.add(new Vertex<>('E'));
        bfsExpected.add(new Vertex<>('C'));
        bfsExpected.add(new Vertex<>('B'));
        bfsExpected.add(new Vertex<>('D'));
        bfsExpected.add(new Vertex<>('A'));
        bfsExpected.add(new Vertex<>('G'));
        bfsExpected.add(new Vertex<>('F'));

        assertEquals(bfsExpected, bfsActual);
    }

    @Test(timeout = TIMEOUT)
    public void bfsUndirectedF() {
        List<Vertex<Character>> bfsActual = GraphAlgorithms.bfs(
                new Vertex<>('F'), undirectedGraph);

        List<Vertex<Character>> bfsExpected = new LinkedList<>();
        bfsExpected.add(new Vertex<>('F'));
        bfsExpected.add(new Vertex<>('D'));
        bfsExpected.add(new Vertex<>('G'));
        bfsExpected.add(new Vertex<>('B'));
        bfsExpected.add(new Vertex<>('E'));
        bfsExpected.add(new Vertex<>('A'));
        bfsExpected.add(new Vertex<>('C'));

        assertEquals(bfsExpected, bfsActual);
    }

    @Test(timeout = TIMEOUT)
    public void bfsUndirectedG() {
        List<Vertex<Character>> bfsActual = GraphAlgorithms.bfs(
                new Vertex<>('G'), undirectedGraph);

        List<Vertex<Character>> bfsExpected = new LinkedList<>();
        bfsExpected.add(new Vertex<>('G'));
        bfsExpected.add(new Vertex<>('A'));
        bfsExpected.add(new Vertex<>('B'));
        bfsExpected.add(new Vertex<>('F'));
        bfsExpected.add(new Vertex<>('C'));
        bfsExpected.add(new Vertex<>('E'));
        bfsExpected.add(new Vertex<>('D'));

        assertEquals(bfsExpected, bfsActual);
    }

    @Test(timeout = TIMEOUT)
    public void bfsUndirectedH() {
        List<Vertex<Character>> bfsActual = GraphAlgorithms.bfs(
                new Vertex<>('H'), undirectedGraph);

        List<Vertex<Character>> bfsExpected = new LinkedList<>();
        bfsExpected.add(new Vertex<>('H'));

        assertEquals(bfsExpected, bfsActual);
    }

    @Test(timeout = TIMEOUT)
    public void bfsDirected1() {
        List<Vertex<Integer>> bfsActual = GraphAlgorithms.bfs(
                new Vertex<>(1), directedGraph);

        List<Vertex<Integer>> bfsExpected = new LinkedList<>();
        int[] sol = {1, 2, 3, 4, 6, 5, 7};
        for (int i: sol) {
            bfsExpected.add(new Vertex<>(i));
        }

        assertEquals(bfsExpected, bfsActual);
    }

    @Test(timeout = TIMEOUT)
    public void bfsDirected2() {
        List<Vertex<Integer>> bfsActual = GraphAlgorithms.bfs(
                new Vertex<>(2), directedGraph);

        List<Vertex<Integer>> bfsExpected = new LinkedList<>();
        int[] sol = {2, 3, 4, 6, 5, 7, 1};
        for (int i: sol) {
            bfsExpected.add(new Vertex<>(i));
        }

        assertEquals(bfsExpected, bfsActual);
    }

    @Test(timeout = TIMEOUT)
    public void bfsDirected3() {
        List<Vertex<Integer>> bfsActual = GraphAlgorithms.bfs(
                new Vertex<>(3), directedGraph);

        List<Vertex<Integer>> bfsExpected = new LinkedList<>();
        int[] sol = {3, 6, 7, 4, 5, 1, 2};
        for (int i: sol) {
            bfsExpected.add(new Vertex<>(i));
        }

        assertEquals(bfsExpected, bfsActual);
    }

    @Test(timeout = TIMEOUT)
    public void bfsDirected4() {
        List<Vertex<Integer>> bfsActual = GraphAlgorithms.bfs(
                new Vertex<>(4), directedGraph);

        List<Vertex<Integer>> bfsExpected = new LinkedList<>();
        int[] sol = {4, 5, 3, 1, 6, 2, 7};
        for (int i: sol) {
            bfsExpected.add(new Vertex<>(i));
        }

        assertEquals(bfsExpected, bfsActual);
    }

    @Test(timeout = TIMEOUT)
    public void bfsDirected5() {
        List<Vertex<Integer>> bfsActual = GraphAlgorithms.bfs(
                new Vertex<>(5), directedGraph);

        List<Vertex<Integer>> bfsExpected = new LinkedList<>();
        int[] sol = {5, 1, 2, 3, 4, 6, 7};
        for (int i: sol) {
            bfsExpected.add(new Vertex<>(i));
        }

        assertEquals(bfsExpected, bfsActual);
    }

    @Test(timeout = TIMEOUT)
    public void bfsDirected6() {
        List<Vertex<Integer>> bfsActual = GraphAlgorithms.bfs(
                new Vertex<>(6), directedGraph);

        List<Vertex<Integer>> bfsExpected = new LinkedList<>();
        int[] sol = {6, 7, 4, 5, 3, 1, 2};
        for (int i: sol) {
            bfsExpected.add(new Vertex<>(i));
        }

        assertEquals(bfsExpected, bfsActual);
    }

    @Test(timeout = TIMEOUT)
    public void bfsDirected7() {
        List<Vertex<Integer>> bfsActual = GraphAlgorithms.bfs(
                new Vertex<>(7), directedGraph);

        List<Vertex<Integer>> bfsExpected = new LinkedList<>();
        int[] sol = {7, 4, 5, 3, 1, 6, 2};
        for (int i: sol) {
            bfsExpected.add(new Vertex<>(i));
        }

        assertEquals(bfsExpected, bfsActual);
    }

    @Test(timeout = TIMEOUT)
    public void dfsUndirectedA() {
        List<Vertex<Character>> dfsActual = GraphAlgorithms.dfs(
                new Vertex<>('A'), undirectedGraph);

        List<Vertex<Character>> dfsExpected = new LinkedList<>();
        dfsExpected.add(new Vertex<>('A'));
        dfsExpected.add(new Vertex<>('G'));
        dfsExpected.add(new Vertex<>('B'));
        dfsExpected.add(new Vertex<>('C'));
        dfsExpected.add(new Vertex<>('E'));
        dfsExpected.add(new Vertex<>('D'));
        dfsExpected.add(new Vertex<>('F'));

        assertEquals(dfsExpected, dfsActual);
    }

    @Test(timeout = TIMEOUT)
    public void dfsUndirectedB() {
        List<Vertex<Character>> dfsActual = GraphAlgorithms.dfs(
                new Vertex<>('B'), undirectedGraph);

        List<Vertex<Character>> dfsExpected = new LinkedList<>();
        dfsExpected.add(new Vertex<>('B'));
        dfsExpected.add(new Vertex<>('C'));
        dfsExpected.add(new Vertex<>('A'));
        dfsExpected.add(new Vertex<>('G'));
        dfsExpected.add(new Vertex<>('F'));
        dfsExpected.add(new Vertex<>('D'));
        dfsExpected.add(new Vertex<>('E'));

        assertEquals(dfsExpected, dfsActual);
    }

    @Test(timeout = TIMEOUT)
    public void dfsUndirectedC() {
        List<Vertex<Character>> dfsActual = GraphAlgorithms.dfs(
                new Vertex<>('C'), undirectedGraph);

        List<Vertex<Character>> dfsExpected = new LinkedList<>();
        dfsExpected.add(new Vertex<>('C'));
        dfsExpected.add(new Vertex<>('A'));
        dfsExpected.add(new Vertex<>('G'));
        dfsExpected.add(new Vertex<>('B'));
        dfsExpected.add(new Vertex<>('E'));
        dfsExpected.add(new Vertex<>('D'));
        dfsExpected.add(new Vertex<>('F'));

        assertEquals(dfsExpected, dfsActual);
    }

    @Test(timeout = TIMEOUT)
    public void dfsUndirectedD() {
        List<Vertex<Character>> dfsActual = GraphAlgorithms.dfs(
                new Vertex<>('D'), undirectedGraph);

        List<Vertex<Character>> dfsExpected = new LinkedList<>();
        dfsExpected.add(new Vertex<>('D'));
        dfsExpected.add(new Vertex<>('B'));
        dfsExpected.add(new Vertex<>('C'));
        dfsExpected.add(new Vertex<>('A'));
        dfsExpected.add(new Vertex<>('G'));
        dfsExpected.add(new Vertex<>('F'));
        dfsExpected.add(new Vertex<>('E'));

        assertEquals(dfsExpected, dfsActual);
    }

    @Test(timeout = TIMEOUT)
    public void dfsUndirectedE() {
        List<Vertex<Character>> dfsActual = GraphAlgorithms.dfs(
                new Vertex<>('E'), undirectedGraph);

        List<Vertex<Character>> dfsExpected = new LinkedList<>();
        dfsExpected.add(new Vertex<>('E'));
        dfsExpected.add(new Vertex<>('C'));
        dfsExpected.add(new Vertex<>('A'));
        dfsExpected.add(new Vertex<>('G'));
        dfsExpected.add(new Vertex<>('B'));
        dfsExpected.add(new Vertex<>('D'));
        dfsExpected.add(new Vertex<>('F'));

        assertEquals(dfsExpected, dfsActual);
    }

    @Test(timeout = TIMEOUT)
    public void dfsUndirectedF() {
        List<Vertex<Character>> dfsActual = GraphAlgorithms.dfs(
                new Vertex<>('F'), undirectedGraph);

        List<Vertex<Character>> dfsExpected = new LinkedList<>();
        dfsExpected.add(new Vertex<>('F'));
        dfsExpected.add(new Vertex<>('D'));
        dfsExpected.add(new Vertex<>('B'));
        dfsExpected.add(new Vertex<>('C'));
        dfsExpected.add(new Vertex<>('A'));
        dfsExpected.add(new Vertex<>('G'));
        dfsExpected.add(new Vertex<>('E'));

        assertEquals(dfsExpected, dfsActual);
    }

    @Test(timeout = TIMEOUT)
    public void dfsUndirectedG() {
        List<Vertex<Character>> dfsActual = GraphAlgorithms.dfs(
                new Vertex<>('G'), undirectedGraph);

        List<Vertex<Character>> dfsExpected = new LinkedList<>();
        dfsExpected.add(new Vertex<>('G'));
        dfsExpected.add(new Vertex<>('A'));
        dfsExpected.add(new Vertex<>('C'));
        dfsExpected.add(new Vertex<>('B'));
        dfsExpected.add(new Vertex<>('E'));
        dfsExpected.add(new Vertex<>('D'));
        dfsExpected.add(new Vertex<>('F'));

        assertEquals(dfsExpected, dfsActual);
    }

    @Test(timeout = TIMEOUT)
    public void dfsUndirectedH() {
        List<Vertex<Character>> dfsActual = GraphAlgorithms.dfs(
                new Vertex<>('H'), undirectedGraph);

        List<Vertex<Character>> dfsExpected = new LinkedList<>();
        dfsExpected.add(new Vertex<>('H'));

        assertEquals(dfsExpected, dfsActual);
    }

    @Test(timeout = TIMEOUT)
    public void dfsDirected1() {
        List<Vertex<Integer>> dfsActual = GraphAlgorithms.dfs(
                new Vertex<>(1), directedGraph);

        List<Vertex<Integer>> dfsExpected = new LinkedList<>();
        int[] sol = {1, 2, 3, 6, 7, 4, 5};
        for (int i: sol) {
            dfsExpected.add(new Vertex<>(i));
        }

        assertEquals(dfsExpected, dfsActual);
    }

    @Test(timeout = TIMEOUT)
    public void dfsDirected2() {
        List<Vertex<Integer>> dfsActual = GraphAlgorithms.dfs(
                new Vertex<>(2), directedGraph);

        List<Vertex<Integer>> dfsExpected = new LinkedList<>();
        int[] sol = {2, 3, 6, 7, 4, 5, 1};
        for (int i: sol) {
            dfsExpected.add(new Vertex<>(i));
        }

        assertEquals(dfsExpected, dfsActual);
    }

    @Test(timeout = TIMEOUT)
    public void dfsDirected3() {
        List<Vertex<Integer>> dfsActual = GraphAlgorithms.dfs(
                new Vertex<>(3), directedGraph);

        List<Vertex<Integer>> dfsExpected = new LinkedList<>();
        int[] sol = {3, 6, 7, 4, 5, 1, 2};
        for (int i: sol) {
            dfsExpected.add(new Vertex<>(i));
        }

        assertEquals(dfsExpected, dfsActual);
    }

    @Test(timeout = TIMEOUT)
    public void dfsDirected4() {
        List<Vertex<Integer>> dfsActual = GraphAlgorithms.dfs(
                new Vertex<>(4), directedGraph);

        List<Vertex<Integer>> dfsExpected = new LinkedList<>();
        int[] sol = {4, 5, 1, 2, 3, 6, 7};
        for (int i: sol) {
            dfsExpected.add(new Vertex<>(i));
        }

        assertEquals(dfsExpected, dfsActual);
    }

    @Test(timeout = TIMEOUT)
    public void dfsDirected5() {
        List<Vertex<Integer>> dfsActual = GraphAlgorithms.dfs(
                new Vertex<>(5), directedGraph);

        List<Vertex<Integer>> dfsExpected = new LinkedList<>();
        int[] sol = {5, 1, 2, 3, 6, 7, 4};
        for (int i: sol) {
            dfsExpected.add(new Vertex<>(i));
        }

        assertEquals(dfsExpected, dfsActual);
    }

    @Test(timeout = TIMEOUT)
    public void dfsDirected6() {
        List<Vertex<Integer>> dfsActual = GraphAlgorithms.dfs(
                new Vertex<>(6), directedGraph);

        List<Vertex<Integer>> dfsExpected = new LinkedList<>();
        int[] sol = {6, 7, 4, 5, 1, 2, 3};
        for (int i: sol) {
            dfsExpected.add(new Vertex<>(i));
        }

        assertEquals(dfsExpected, dfsActual);
    }

    @Test(timeout = TIMEOUT)
    public void dfsDirected7() {
        List<Vertex<Integer>> dfsActual = GraphAlgorithms.dfs(
                new Vertex<>(7), directedGraph);

        List<Vertex<Integer>> dfsExpected = new LinkedList<>();
        int[] sol = {7, 4, 5, 1, 2, 3, 6};
        for (int i: sol) {
            dfsExpected.add(new Vertex<>(i));
        }

        assertEquals(dfsExpected, dfsActual);
    }

    @Test(timeout = TIMEOUT)
    public void dijkstrasUndirectedA() {
        Map<Vertex<Character>, Integer> dijkActual = GraphAlgorithms.dijkstras(
                new Vertex<Character>('A'), undirectedGraph);
        Map<Vertex<Character>, Integer> dijkExpected = new HashMap<>();
        dijkExpected.put(new Vertex<>('A'), 0);
        dijkExpected.put(new Vertex<>('B'), 6);
        dijkExpected.put(new Vertex<>('C'), 2);
        dijkExpected.put(new Vertex<>('D'), 8);
        dijkExpected.put(new Vertex<>('E'), 7);
        dijkExpected.put(new Vertex<>('F'), 11);
        dijkExpected.put(new Vertex<>('G'), 7);
        dijkExpected.put(new Vertex<>('H'), Integer.MAX_VALUE);

        assertEquals(dijkExpected, dijkActual);
    }

    @Test(timeout = TIMEOUT)
    public void dijkstrasUndirectedB() {
        Map<Vertex<Character>, Integer> dijkActual = GraphAlgorithms.dijkstras(
                new Vertex<Character>('B'), undirectedGraph);
        Map<Vertex<Character>, Integer> dijkExpected = new HashMap<>();
        dijkExpected.put(new Vertex<>('A'), 6);
        dijkExpected.put(new Vertex<>('B'), 0);
        dijkExpected.put(new Vertex<>('C'), 4);
        dijkExpected.put(new Vertex<>('D'), 2);
        dijkExpected.put(new Vertex<>('E'), 1);
        dijkExpected.put(new Vertex<>('F'), 5);
        dijkExpected.put(new Vertex<>('G'), 1);
        dijkExpected.put(new Vertex<>('H'), Integer.MAX_VALUE);

        assertEquals(dijkExpected, dijkActual);
    }

    @Test(timeout = TIMEOUT)
    public void dijkstrasUndirectedC() {
        Map<Vertex<Character>, Integer> dijkActual = GraphAlgorithms.dijkstras(
                new Vertex<Character>('C'), undirectedGraph);
        Map<Vertex<Character>, Integer> dijkExpected = new HashMap<>();
        dijkExpected.put(new Vertex<>('A'), 2);
        dijkExpected.put(new Vertex<>('B'), 4);
        dijkExpected.put(new Vertex<>('C'), 0);
        dijkExpected.put(new Vertex<>('D'), 6);
        dijkExpected.put(new Vertex<>('E'), 5);
        dijkExpected.put(new Vertex<>('F'), 9);
        dijkExpected.put(new Vertex<>('G'), 5);
        dijkExpected.put(new Vertex<>('H'), Integer.MAX_VALUE);

        assertEquals(dijkExpected, dijkActual);
    }

    @Test(timeout = TIMEOUT)
    public void dijkstrasUndirectedD() {
        Map<Vertex<Character>, Integer> dijkActual = GraphAlgorithms.dijkstras(
                new Vertex<Character>('D'), undirectedGraph);
        Map<Vertex<Character>, Integer> dijkExpected = new HashMap<>();
        dijkExpected.put(new Vertex<>('A'), 8);
        dijkExpected.put(new Vertex<>('B'), 2);
        dijkExpected.put(new Vertex<>('C'), 6);
        dijkExpected.put(new Vertex<>('D'), 0);
        dijkExpected.put(new Vertex<>('E'), 3);
        dijkExpected.put(new Vertex<>('F'), 3);
        dijkExpected.put(new Vertex<>('G'), 3);
        dijkExpected.put(new Vertex<>('H'), Integer.MAX_VALUE);

        assertEquals(dijkExpected, dijkActual);
    }

    @Test(timeout = TIMEOUT)
    public void dijkstrasUndirectedE() {
        Map<Vertex<Character>, Integer> dijkActual = GraphAlgorithms.dijkstras(
                new Vertex<Character>('E'), undirectedGraph);
        Map<Vertex<Character>, Integer> dijkExpected = new HashMap<>();
        dijkExpected.put(new Vertex<>('A'), 7);
        dijkExpected.put(new Vertex<>('B'), 1);
        dijkExpected.put(new Vertex<>('C'), 5);
        dijkExpected.put(new Vertex<>('D'), 3);
        dijkExpected.put(new Vertex<>('E'), 0);
        dijkExpected.put(new Vertex<>('F'), 6);
        dijkExpected.put(new Vertex<>('G'), 2);
        dijkExpected.put(new Vertex<>('H'), Integer.MAX_VALUE);

        assertEquals(dijkExpected, dijkActual);
    }

    @Test(timeout = TIMEOUT)
    public void dijkstrasUndirectedF() {
        Map<Vertex<Character>, Integer> dijkActual = GraphAlgorithms.dijkstras(
                new Vertex<Character>('F'), undirectedGraph);
        Map<Vertex<Character>, Integer> dijkExpected = new HashMap<>();
        dijkExpected.put(new Vertex<>('A'), 11);
        dijkExpected.put(new Vertex<>('B'), 5);
        dijkExpected.put(new Vertex<>('C'), 9);
        dijkExpected.put(new Vertex<>('D'), 3);
        dijkExpected.put(new Vertex<>('E'), 6);
        dijkExpected.put(new Vertex<>('F'), 0);
        dijkExpected.put(new Vertex<>('G'), 6);
        dijkExpected.put(new Vertex<>('H'), Integer.MAX_VALUE);

        assertEquals(dijkExpected, dijkActual);
    }

    @Test(timeout = TIMEOUT)
    public void dijkstrasUndirectedG() {
        Map<Vertex<Character>, Integer> dijkActual = GraphAlgorithms.dijkstras(
                new Vertex<Character>('G'), undirectedGraph);
        Map<Vertex<Character>, Integer> dijkExpected = new HashMap<>();
        dijkExpected.put(new Vertex<>('A'), 7);
        dijkExpected.put(new Vertex<>('B'), 1);
        dijkExpected.put(new Vertex<>('C'), 5);
        dijkExpected.put(new Vertex<>('D'), 3);
        dijkExpected.put(new Vertex<>('E'), 2);
        dijkExpected.put(new Vertex<>('F'), 6);
        dijkExpected.put(new Vertex<>('G'), 0);
        dijkExpected.put(new Vertex<>('H'), Integer.MAX_VALUE);

        assertEquals(dijkExpected, dijkActual);
    }

    @Test(timeout = TIMEOUT)
    public void dijkstrasUndirectedH() {
        Map<Vertex<Character>, Integer> dijkActual = GraphAlgorithms.dijkstras(
                new Vertex<Character>('H'), undirectedGraph);
        Map<Vertex<Character>, Integer> dijkExpected = new HashMap<>();
        dijkExpected.put(new Vertex<>('A'), Integer.MAX_VALUE);
        dijkExpected.put(new Vertex<>('B'), Integer.MAX_VALUE);
        dijkExpected.put(new Vertex<>('C'), Integer.MAX_VALUE);
        dijkExpected.put(new Vertex<>('D'), Integer.MAX_VALUE);
        dijkExpected.put(new Vertex<>('E'), Integer.MAX_VALUE);
        dijkExpected.put(new Vertex<>('F'), Integer.MAX_VALUE);
        dijkExpected.put(new Vertex<>('G'), Integer.MAX_VALUE);
        dijkExpected.put(new Vertex<>('H'), 0);

        assertEquals(dijkExpected, dijkActual);
    }

    @Test(timeout = TIMEOUT)
    public void dijkstrasDirected1() {
        Map<Vertex<Integer>, Integer> dijkActual = GraphAlgorithms.dijkstras(
                new Vertex<Integer>(1), directedGraph);
        Map<Vertex<Integer>, Integer> dijkExpected = new HashMap<>();
        dijkExpected.put(new Vertex<>(1), 0);
        dijkExpected.put(new Vertex<>(2), 3);
        dijkExpected.put(new Vertex<>(3), 4);
        dijkExpected.put(new Vertex<>(4), 8);
        dijkExpected.put(new Vertex<>(5), 12);
        dijkExpected.put(new Vertex<>(6), 12);
        dijkExpected.put(new Vertex<>(7), 24);

        assertEquals(dijkExpected, dijkActual);
    }

    @Test(timeout = TIMEOUT)
    public void dijkstrasDirected2() {
        Map<Vertex<Integer>, Integer> dijkActual = GraphAlgorithms.dijkstras(
                new Vertex<Integer>(2), directedGraph);
        Map<Vertex<Integer>, Integer> dijkExpected = new HashMap<>();
        dijkExpected.put(new Vertex<>(1), 11);
        dijkExpected.put(new Vertex<>(2), 0);
        dijkExpected.put(new Vertex<>(3), 1);
        dijkExpected.put(new Vertex<>(4), 5);
        dijkExpected.put(new Vertex<>(5), 9);
        dijkExpected.put(new Vertex<>(6), 9);
        dijkExpected.put(new Vertex<>(7), 21);

        assertEquals(dijkExpected, dijkActual);
    }

    @Test(timeout = TIMEOUT)
    public void dijkstrasDirected3() {
        Map<Vertex<Integer>, Integer> dijkActual = GraphAlgorithms.dijkstras(
                new Vertex<Integer>(3), directedGraph);
        Map<Vertex<Integer>, Integer> dijkExpected = new HashMap<>();
        dijkExpected.put(new Vertex<>(1), 28);
        dijkExpected.put(new Vertex<>(2), 31);
        dijkExpected.put(new Vertex<>(3), 0);
        dijkExpected.put(new Vertex<>(4), 22);
        dijkExpected.put(new Vertex<>(5), 26);
        dijkExpected.put(new Vertex<>(6), 8);
        dijkExpected.put(new Vertex<>(7), 20);

        assertEquals(dijkExpected, dijkActual);
    }

    @Test(timeout = TIMEOUT)
    public void dijkstrasDirected4() {
        Map<Vertex<Integer>, Integer> dijkActual = GraphAlgorithms.dijkstras(
                new Vertex<Integer>(4), directedGraph);
        Map<Vertex<Integer>, Integer> dijkExpected = new HashMap<>();
        dijkExpected.put(new Vertex<>(1), 6);
        dijkExpected.put(new Vertex<>(2), 9);
        dijkExpected.put(new Vertex<>(3), 10);
        dijkExpected.put(new Vertex<>(4), 0);
        dijkExpected.put(new Vertex<>(5), 4);
        dijkExpected.put(new Vertex<>(6), 18);
        dijkExpected.put(new Vertex<>(7), 30);

        assertEquals(dijkExpected, dijkActual);
    }

    @Test(timeout = TIMEOUT)
    public void dijkstrasDirected5() {
        Map<Vertex<Integer>, Integer> dijkActual = GraphAlgorithms.dijkstras(
                new Vertex<Integer>(5), directedGraph);
        Map<Vertex<Integer>, Integer> dijkExpected = new HashMap<>();
        dijkExpected.put(new Vertex<>(1), 2);
        dijkExpected.put(new Vertex<>(2), 5);
        dijkExpected.put(new Vertex<>(3), 6);
        dijkExpected.put(new Vertex<>(4), 10);
        dijkExpected.put(new Vertex<>(5), 0);
        dijkExpected.put(new Vertex<>(6), 14);
        dijkExpected.put(new Vertex<>(7), 26);

        assertEquals(dijkExpected, dijkActual);
    }

    @Test(timeout = TIMEOUT)
    public void dijkstrasDirected6() {
        Map<Vertex<Integer>, Integer> dijkActual = GraphAlgorithms.dijkstras(
                new Vertex<Integer>(6), directedGraph);
        Map<Vertex<Integer>, Integer> dijkExpected = new HashMap<>();
        dijkExpected.put(new Vertex<>(1), 20);
        dijkExpected.put(new Vertex<>(2), 23);
        dijkExpected.put(new Vertex<>(3), 24);
        dijkExpected.put(new Vertex<>(4), 14);
        dijkExpected.put(new Vertex<>(5), 18);
        dijkExpected.put(new Vertex<>(6), 0);
        dijkExpected.put(new Vertex<>(7), 12);

        assertEquals(dijkExpected, dijkActual);
    }

    @Test(timeout = TIMEOUT)
    public void dijkstrasDirected7() {
        Map<Vertex<Integer>, Integer> dijkActual = GraphAlgorithms.dijkstras(
                new Vertex<Integer>(7), directedGraph);
        Map<Vertex<Integer>, Integer> dijkExpected = new HashMap<>();
        dijkExpected.put(new Vertex<>(1), 8);
        dijkExpected.put(new Vertex<>(2), 11);
        dijkExpected.put(new Vertex<>(3), 12);
        dijkExpected.put(new Vertex<>(4), 2);
        dijkExpected.put(new Vertex<>(5), 6);
        dijkExpected.put(new Vertex<>(6), 20);
        dijkExpected.put(new Vertex<>(7), 0);

        assertEquals(dijkExpected, dijkActual);
    }

    @Test(timeout = TIMEOUT)
    public void primsUndirected() {
        Set<Vertex<Character>> vertices = new HashSet<>();
        for (int i = 65; i <= 71; i++) {
            vertices.add(new Vertex<Character>((char) i));
        }

        Set<Edge<Character>> edges = new LinkedHashSet<>();
        edges.add(new Edge<>(new Vertex<>('A'), new Vertex<>('G'), 18));
        edges.add(new Edge<>(new Vertex<>('G'), new Vertex<>('A'), 18));

        edges.add(new Edge<>(new Vertex<>('A'), new Vertex<>('C'), 2));
        edges.add(new Edge<>(new Vertex<>('C'), new Vertex<>('A'), 2));

        edges.add(new Edge<>(new Vertex<>('C'), new Vertex<>('B'), 4));
        edges.add(new Edge<>(new Vertex<>('B'), new Vertex<>('C'), 4));

        edges.add(new Edge<>(new Vertex<>('C'), new Vertex<>('E'), 5));
        edges.add(new Edge<>(new Vertex<>('E'), new Vertex<>('C'), 5));

        edges.add(new Edge<>(new Vertex<>('E'), new Vertex<>('B'), 1));
        edges.add(new Edge<>(new Vertex<>('B'), new Vertex<>('E'), 1));

        edges.add(new Edge<>(new Vertex<>('B'), new Vertex<>('D'), 2));
        edges.add(new Edge<>(new Vertex<>('D'), new Vertex<>('B'), 2));

        edges.add(new Edge<>(new Vertex<>('E'), new Vertex<>('D'), 7));
        edges.add(new Edge<>(new Vertex<>('D'), new Vertex<>('E'), 7));

        edges.add(new Edge<>(new Vertex<>('F'), new Vertex<>('D'), 3));
        edges.add(new Edge<>(new Vertex<>('D'), new Vertex<>('F'), 3));

        edges.add(new Edge<>(new Vertex<>('B'), new Vertex<>('G'), 1));
        edges.add(new Edge<>(new Vertex<>('G'), new Vertex<>('B'), 1));

        edges.add(new Edge<>(new Vertex<>('F'), new Vertex<>('G'), 6));
        edges.add(new Edge<>(new Vertex<>('G'), new Vertex<>('F'), 6));

        Set<Edge<Character>> mstActual = GraphAlgorithms.prims(
                new Vertex<>('A'), new Graph<>(vertices, edges));
        Set<Edge<Character>> mstExpected = new HashSet<>();
        mstExpected.add(new Edge<>(new Vertex<>('A'), new Vertex<>('C'), 2));
        mstExpected.add(new Edge<>(new Vertex<>('C'), new Vertex<>('A'), 2));

        mstExpected.add(new Edge<>(new Vertex<>('D'), new Vertex<>('F'), 3));
        mstExpected.add(new Edge<>(new Vertex<>('F'), new Vertex<>('D'), 3));

        mstExpected.add(new Edge<>(new Vertex<>('B'), new Vertex<>('G'), 1));
        mstExpected.add(new Edge<>(new Vertex<>('G'), new Vertex<>('B'), 1));

        mstExpected.add(new Edge<>(new Vertex<>('B'), new Vertex<>('D'), 2));
        mstExpected.add(new Edge<>(new Vertex<>('D'), new Vertex<>('B'), 2));

        mstExpected.add(new Edge<>(new Vertex<>('B'), new Vertex<>('D'), 2));
        mstExpected.add(new Edge<>(new Vertex<>('D'), new Vertex<>('B'), 2));

        mstExpected.add(new Edge<>(new Vertex<>('C'), new Vertex<>('B'), 4));
        mstExpected.add(new Edge<>(new Vertex<>('B'), new Vertex<>('C'), 4));

        mstExpected.add(new Edge<>(new Vertex<>('B'), new Vertex<>('E'), 1));
        mstExpected.add(new Edge<>(new Vertex<>('E'), new Vertex<>('B'), 1));

        assertEquals(mstExpected, mstActual);
    }

    @Test(timeout = TIMEOUT)
    public void primsNullCase() {
        Set<Edge<Character>> edges = new HashSet<>();
        edges.add(new Edge<>(new Vertex<>('A'), new Vertex<>('B'), 0));
        Set<Vertex<Character>> vertices = new HashSet<>();
        vertices.add(new Vertex<>('A'));
        vertices.add(new Vertex<>('B'));
        vertices.add(new Vertex<>('C'));

        Set<Edge<Character>> mstActual = GraphAlgorithms.prims(
                new Vertex<>('A'), new Graph<>(vertices, edges));

        assertNull(mstActual); //Read second sentence of Prims Javadocs (C is disconnected)
    }
}
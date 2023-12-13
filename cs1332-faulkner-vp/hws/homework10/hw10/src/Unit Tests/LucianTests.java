import org.junit.Before;
import org.junit.Test;

import java.util.*;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;

/**
 * Graph algorithm JUnit Tests
 *
 * @author Lucian Tash
 * @version 1.0
 */
public class LucianTests {

    private Graph<Character> graph; // undirected |v|=8
    private Graph<Character> largeGraph; // undirected |v|=26
    private Graph<Character> directedGraph; // directed |v|=8
    private Graph<Character> connectedGraph; // undirected |v|=7 (for Prim's)
    private Graph<Character> largeConnectedGraph; // undirected |v|=15 (for Prim's)
    public static final int TIMEOUT = 200;

    @Before
    public void start() {
        graph = createUndirectedGraph();
        largeGraph = createLargeUndirectedGraph();
        directedGraph = createDirectedGraph();
        connectedGraph = createConnectedGraph();
        largeConnectedGraph = createLargeConnectedGraph();
    }


    /**
     *          A —— B
     *          |    |   G       <--- G is not connected to graph
     *      F — C —— D
     *           \  / \
     *            E    H
     *
     *  8 vertices, undirected
     *  See image here: https://imgur.com/a/7JlpQh5
     */
    private Graph<Character> createUndirectedGraph() {
        Set<Vertex<Character>> vertices = new HashSet<>();
        vertices.add(new Vertex<>('A'));
        vertices.add(new Vertex<>('B'));
        vertices.add(new Vertex<>('C'));
        vertices.add(new Vertex<>('D'));
        vertices.add(new Vertex<>('E'));
        vertices.add(new Vertex<>('F'));
        vertices.add(new Vertex<>('G'));
        vertices.add(new Vertex<>('H'));

        Set<Edge<Character>> edges = new LinkedHashSet<>();
        edges.add(new Edge<>(new Vertex<>('A'), new Vertex<>('B'), 1));
        edges.add(new Edge<>(new Vertex<>('A'), new Vertex<>('C'), 7));
        edges.add(new Edge<>(new Vertex<>('B'), new Vertex<>('A'), 1));
        edges.add(new Edge<>(new Vertex<>('B'), new Vertex<>('D'), 6));
        edges.add(new Edge<>(new Vertex<>('C'), new Vertex<>('A'), 7));
        edges.add(new Edge<>(new Vertex<>('C'), new Vertex<>('D'), 10));
        edges.add(new Edge<>(new Vertex<>('C'), new Vertex<>('E'), 2));
        edges.add(new Edge<>(new Vertex<>('C'), new Vertex<>('F'), 5));
        edges.add(new Edge<>(new Vertex<>('D'), new Vertex<>('B'), 6));
        edges.add(new Edge<>(new Vertex<>('D'), new Vertex<>('C'), 10));
        edges.add(new Edge<>(new Vertex<>('D'), new Vertex<>('E'), 3));
        edges.add(new Edge<>(new Vertex<>('D'), new Vertex<>('H'), 4));
        edges.add(new Edge<>(new Vertex<>('E'), new Vertex<>('C'), 2));
        edges.add(new Edge<>(new Vertex<>('E'), new Vertex<>('D'), 3));
        edges.add(new Edge<>(new Vertex<>('F'), new Vertex<>('C'), 5));
        edges.add(new Edge<>(new Vertex<>('H'), new Vertex<>('D'), 4));

        return new Graph<>(vertices, edges);
    }

    /**
     * 7 vertices, undirected
     * See image here: https://imgur.com/a/qov6Qsy
     */
    private Graph<Character> createConnectedGraph() {
        Set<Vertex<Character>> vertices = new HashSet<>();
        vertices.add(new Vertex<>('A'));
        vertices.add(new Vertex<>('B'));
        vertices.add(new Vertex<>('C'));
        vertices.add(new Vertex<>('D'));
        vertices.add(new Vertex<>('E'));
        vertices.add(new Vertex<>('F'));
        vertices.add(new Vertex<>('G'));

        Set<Edge<Character>> edges = new LinkedHashSet<>();
        edges.add(new Edge<>(new Vertex<>('A'), new Vertex<>('B'), 1));
        edges.add(new Edge<>(new Vertex<>('A'), new Vertex<>('C'), 7));
        edges.add(new Edge<>(new Vertex<>('B'), new Vertex<>('A'), 1));
        edges.add(new Edge<>(new Vertex<>('B'), new Vertex<>('D'), 6));
        edges.add(new Edge<>(new Vertex<>('C'), new Vertex<>('A'), 7));
        edges.add(new Edge<>(new Vertex<>('C'), new Vertex<>('D'), 10));
        edges.add(new Edge<>(new Vertex<>('C'), new Vertex<>('E'), 2));
        edges.add(new Edge<>(new Vertex<>('C'), new Vertex<>('F'), 5));
        edges.add(new Edge<>(new Vertex<>('D'), new Vertex<>('B'), 6));
        edges.add(new Edge<>(new Vertex<>('D'), new Vertex<>('C'), 10));
        edges.add(new Edge<>(new Vertex<>('D'), new Vertex<>('E'), 3));
        edges.add(new Edge<>(new Vertex<>('D'), new Vertex<>('G'), 4));
        edges.add(new Edge<>(new Vertex<>('E'), new Vertex<>('C'), 2));
        edges.add(new Edge<>(new Vertex<>('E'), new Vertex<>('D'), 3));
        edges.add(new Edge<>(new Vertex<>('F'), new Vertex<>('C'), 5));
        edges.add(new Edge<>(new Vertex<>('G'), new Vertex<>('D'), 4));

        return new Graph<>(vertices, edges);
    }

    /**
     *  8 vertices, directed
     *  See image here: https://imgur.com/a/oNOygQ9
     */
    private Graph<Character> createDirectedGraph() {
        Set<Vertex<Character>> vertices = new HashSet<>();
        vertices.add(new Vertex<>('A'));
        vertices.add(new Vertex<>('B'));
        vertices.add(new Vertex<>('C'));
        vertices.add(new Vertex<>('D'));
        vertices.add(new Vertex<>('E'));
        vertices.add(new Vertex<>('F'));
        vertices.add(new Vertex<>('G'));
        vertices.add(new Vertex<>('H'));

        Set<Edge<Character>> edges = new LinkedHashSet<>();
        edges.add(new Edge<>(new Vertex<>('A'), new Vertex<>('G'), 8));
        edges.add(new Edge<>(new Vertex<>('C'), new Vertex<>('F'), 4));
        edges.add(new Edge<>(new Vertex<>('D'), new Vertex<>('G'), 2));
        edges.add(new Edge<>(new Vertex<>('E'), new Vertex<>('H'), 0));
        edges.add(new Edge<>(new Vertex<>('F'), new Vertex<>('G'), 10));
        edges.add(new Edge<>(new Vertex<>('G'), new Vertex<>('C'), 6));
        edges.add(new Edge<>(new Vertex<>('G'), new Vertex<>('D'), 2));
        edges.add(new Edge<>(new Vertex<>('G'), new Vertex<>('H'), 3));
        edges.add(new Edge<>(new Vertex<>('H'), new Vertex<>('A'), 4));
        edges.add(new Edge<>(new Vertex<>('H'), new Vertex<>('E'), 0));

        return new Graph<>(vertices, edges);
    }

    /**
     * 26 vertices, undirected
     * See image here: https://imgur.com/a/DOBL8ug
     */
    private Graph<Character> createLargeUndirectedGraph() {
        Set<Vertex<Character>> vertices = new HashSet<>();
        for (int i = 0; i < 26; i++) {
            vertices.add(new Vertex<>((char) ('A' + i))); // A to Z
        }

        Set<Edge<Character>> edges = new LinkedHashSet<>();
        edges.add(new Edge<>(new Vertex<>('A'), new Vertex<>('B'), 0));
        edges.add(new Edge<>(new Vertex<>('A'), new Vertex<>('C'), 2));
        edges.add(new Edge<>(new Vertex<>('A'), new Vertex<>('J'), 7));
        edges.add(new Edge<>(new Vertex<>('A'), new Vertex<>('V'), 3));
        edges.add(new Edge<>(new Vertex<>('B'), new Vertex<>('A'), 0));
        edges.add(new Edge<>(new Vertex<>('B'), new Vertex<>('Y'), 1));
        edges.add(new Edge<>(new Vertex<>('B'), new Vertex<>('Z'), 4));
        edges.add(new Edge<>(new Vertex<>('C'), new Vertex<>('A'), 2));
        edges.add(new Edge<>(new Vertex<>('C'), new Vertex<>('L'), 0));
        edges.add(new Edge<>(new Vertex<>('C'), new Vertex<>('Z'), 3));
        edges.add(new Edge<>(new Vertex<>('D'), new Vertex<>('M'), 3));
        edges.add(new Edge<>(new Vertex<>('D'), new Vertex<>('O'), 2));
        edges.add(new Edge<>(new Vertex<>('F'), new Vertex<>('M'), 7));
        edges.add(new Edge<>(new Vertex<>('F'), new Vertex<>('O'), 1));
        edges.add(new Edge<>(new Vertex<>('G'), new Vertex<>('T'), 3));
        edges.add(new Edge<>(new Vertex<>('G'), new Vertex<>('W'), 3));
        edges.add(new Edge<>(new Vertex<>('H'), new Vertex<>('M'), 6));
        edges.add(new Edge<>(new Vertex<>('I'), new Vertex<>('Q'), 6));
        edges.add(new Edge<>(new Vertex<>('I'), new Vertex<>('X'), 7));
        edges.add(new Edge<>(new Vertex<>('J'), new Vertex<>('A'), 7));
        edges.add(new Edge<>(new Vertex<>('J'), new Vertex<>('N'), 6));
        edges.add(new Edge<>(new Vertex<>('J'), new Vertex<>('U'), 1));
        edges.add(new Edge<>(new Vertex<>('K'), new Vertex<>('R'), 9));
        edges.add(new Edge<>(new Vertex<>('L'), new Vertex<>('C'), 0));
        edges.add(new Edge<>(new Vertex<>('M'), new Vertex<>('D'), 3));
        edges.add(new Edge<>(new Vertex<>('M'), new Vertex<>('F'), 7));
        edges.add(new Edge<>(new Vertex<>('M'), new Vertex<>('H'), 6));
        edges.add(new Edge<>(new Vertex<>('M'), new Vertex<>('O'), 8));
        edges.add(new Edge<>(new Vertex<>('N'), new Vertex<>('J'), 6));
        edges.add(new Edge<>(new Vertex<>('N'), new Vertex<>('S'), 7));
        edges.add(new Edge<>(new Vertex<>('N'), new Vertex<>('Y'), 9));
        edges.add(new Edge<>(new Vertex<>('O'), new Vertex<>('D'), 2));
        edges.add(new Edge<>(new Vertex<>('O'), new Vertex<>('F'), 1));
        edges.add(new Edge<>(new Vertex<>('O'), new Vertex<>('M'), 8));
        edges.add(new Edge<>(new Vertex<>('O'), new Vertex<>('Q'), 4));
        edges.add(new Edge<>(new Vertex<>('P'), new Vertex<>('T'), 2));
        edges.add(new Edge<>(new Vertex<>('P'), new Vertex<>('U'), 4));
        edges.add(new Edge<>(new Vertex<>('P'), new Vertex<>('W'), 6));
        edges.add(new Edge<>(new Vertex<>('Q'), new Vertex<>('I'), 6));
        edges.add(new Edge<>(new Vertex<>('Q'), new Vertex<>('O'), 4));
        edges.add(new Edge<>(new Vertex<>('Q'), new Vertex<>('X'), 5));
        edges.add(new Edge<>(new Vertex<>('R'), new Vertex<>('K'), 9));
        edges.add(new Edge<>(new Vertex<>('S'), new Vertex<>('N'), 7));
        edges.add(new Edge<>(new Vertex<>('T'), new Vertex<>('G'), 3));
        edges.add(new Edge<>(new Vertex<>('T'), new Vertex<>('P'), 2));
        edges.add(new Edge<>(new Vertex<>('U'), new Vertex<>('J'), 1));
        edges.add(new Edge<>(new Vertex<>('U'), new Vertex<>('P'), 4));
        edges.add(new Edge<>(new Vertex<>('V'), new Vertex<>('A'), 3));
        edges.add(new Edge<>(new Vertex<>('W'), new Vertex<>('G'), 3));
        edges.add(new Edge<>(new Vertex<>('W'), new Vertex<>('P'), 6));
        edges.add(new Edge<>(new Vertex<>('X'), new Vertex<>('I'), 7));
        edges.add(new Edge<>(new Vertex<>('X'), new Vertex<>('Q'), 5));
        edges.add(new Edge<>(new Vertex<>('Y'), new Vertex<>('B'), 1));
        edges.add(new Edge<>(new Vertex<>('Y'), new Vertex<>('N'), 9));
        edges.add(new Edge<>(new Vertex<>('Y'), new Vertex<>('Z'), 8));
        edges.add(new Edge<>(new Vertex<>('Z'), new Vertex<>('B'), 4));
        edges.add(new Edge<>(new Vertex<>('Z'), new Vertex<>('C'), 3));
        edges.add(new Edge<>(new Vertex<>('Z'), new Vertex<>('Y'), 8));

        return new Graph<>(vertices, edges);
    }

    /**
     * 15 vertices, undirected
     * See image here: https://imgur.com/a/XaeXISY
     */
    private Graph<Character> createLargeConnectedGraph() {
        Set<Vertex<Character>> vertices = new HashSet<>();
        vertices.add(new Vertex<>('A'));
        vertices.add(new Vertex<>('B'));
        vertices.add(new Vertex<>('C'));
        vertices.add(new Vertex<>('G'));
        vertices.add(new Vertex<>('J'));
        vertices.add(new Vertex<>('L'));
        vertices.add(new Vertex<>('N'));
        vertices.add(new Vertex<>('P'));
        vertices.add(new Vertex<>('S'));
        vertices.add(new Vertex<>('T'));
        vertices.add(new Vertex<>('U'));
        vertices.add(new Vertex<>('V'));
        vertices.add(new Vertex<>('W'));
        vertices.add(new Vertex<>('Y'));
        vertices.add(new Vertex<>('Z'));

        Set<Edge<Character>> edges = new LinkedHashSet<>();
        edges.add(new Edge<>(new Vertex<>('A'), new Vertex<>('B'), 0));
        edges.add(new Edge<>(new Vertex<>('A'), new Vertex<>('C'), 2));
        edges.add(new Edge<>(new Vertex<>('A'), new Vertex<>('J'), 7));
        edges.add(new Edge<>(new Vertex<>('A'), new Vertex<>('V'), 3));
        edges.add(new Edge<>(new Vertex<>('B'), new Vertex<>('A'), 0));
        edges.add(new Edge<>(new Vertex<>('B'), new Vertex<>('Y'), 1));
        edges.add(new Edge<>(new Vertex<>('B'), new Vertex<>('Z'), 4));
        edges.add(new Edge<>(new Vertex<>('C'), new Vertex<>('A'), 2));
        edges.add(new Edge<>(new Vertex<>('C'), new Vertex<>('L'), 0));
        edges.add(new Edge<>(new Vertex<>('C'), new Vertex<>('Z'), 3));
        edges.add(new Edge<>(new Vertex<>('G'), new Vertex<>('T'), 3));
        edges.add(new Edge<>(new Vertex<>('G'), new Vertex<>('W'), 3));
        edges.add(new Edge<>(new Vertex<>('J'), new Vertex<>('A'), 7));
        edges.add(new Edge<>(new Vertex<>('J'), new Vertex<>('N'), 6));
        edges.add(new Edge<>(new Vertex<>('J'), new Vertex<>('U'), 1));
        edges.add(new Edge<>(new Vertex<>('L'), new Vertex<>('C'), 0));
        edges.add(new Edge<>(new Vertex<>('N'), new Vertex<>('J'), 6));
        edges.add(new Edge<>(new Vertex<>('N'), new Vertex<>('S'), 7));
        edges.add(new Edge<>(new Vertex<>('N'), new Vertex<>('Y'), 9));
        edges.add(new Edge<>(new Vertex<>('P'), new Vertex<>('T'), 2));
        edges.add(new Edge<>(new Vertex<>('P'), new Vertex<>('U'), 4));
        edges.add(new Edge<>(new Vertex<>('P'), new Vertex<>('W'), 6));
        edges.add(new Edge<>(new Vertex<>('S'), new Vertex<>('N'), 7));
        edges.add(new Edge<>(new Vertex<>('T'), new Vertex<>('G'), 3));
        edges.add(new Edge<>(new Vertex<>('T'), new Vertex<>('P'), 2));
        edges.add(new Edge<>(new Vertex<>('U'), new Vertex<>('J'), 1));
        edges.add(new Edge<>(new Vertex<>('U'), new Vertex<>('P'), 4));
        edges.add(new Edge<>(new Vertex<>('V'), new Vertex<>('A'), 3));
        edges.add(new Edge<>(new Vertex<>('W'), new Vertex<>('G'), 3));
        edges.add(new Edge<>(new Vertex<>('W'), new Vertex<>('P'), 6));
        edges.add(new Edge<>(new Vertex<>('Y'), new Vertex<>('B'), 1));
        edges.add(new Edge<>(new Vertex<>('Y'), new Vertex<>('N'), 9));
        edges.add(new Edge<>(new Vertex<>('Y'), new Vertex<>('Z'), 8));
        edges.add(new Edge<>(new Vertex<>('Z'), new Vertex<>('B'), 4));
        edges.add(new Edge<>(new Vertex<>('Z'), new Vertex<>('C'), 3));
        edges.add(new Edge<>(new Vertex<>('Z'), new Vertex<>('Y'), 8));
        return new Graph<>(vertices, edges);
    }

    @Test(timeout = TIMEOUT, expected = IllegalArgumentException.class)
    public void testDFSStartNotInGraph() {
        GraphAlgorithms.dfs(new Vertex<>('L'), graph);
    }

    @Test(timeout = TIMEOUT, expected = IllegalArgumentException.class)
    public void testDFSNullGraph() {
        GraphAlgorithms.dfs(new Vertex<>('A'), null);
    }

    @Test(timeout = TIMEOUT, expected = IllegalArgumentException.class)
    public void testDFSNullStart() {
        GraphAlgorithms.dfs(null, graph);
    }

    @Test(timeout = TIMEOUT, expected = IllegalArgumentException.class)
    public void testBFSStartNotInGraph() {
        GraphAlgorithms.bfs(new Vertex<>('L'), graph);
    }

    @Test(timeout = TIMEOUT, expected = IllegalArgumentException.class)
    public void testBFSNullGraph() {
        GraphAlgorithms.bfs(new Vertex<>('A'), null);
    }

    @Test(timeout = TIMEOUT, expected = IllegalArgumentException.class)
    public void testBFSNullStart() {
        GraphAlgorithms.bfs(null, graph);
    }

    @Test(timeout = TIMEOUT, expected = IllegalArgumentException.class)
    public void testDijkstrasStartNotInGraph() {
        GraphAlgorithms.dijkstras(new Vertex<>('L'), graph);
    }

    @Test(timeout = TIMEOUT, expected = IllegalArgumentException.class)
    public void testDijkstrasNullGraph() {
        GraphAlgorithms.dijkstras(new Vertex<>('A'), null);
    }

    @Test(timeout = TIMEOUT, expected = IllegalArgumentException.class)
    public void testDijkstrasNullStart() {
        GraphAlgorithms.dijkstras(null, graph);
    }

    @Test(timeout = TIMEOUT, expected = IllegalArgumentException.class)
    public void testPrimsStartNotInGraph() {
        GraphAlgorithms.prims(new Vertex<>('L'), graph);
    }

    @Test(timeout = TIMEOUT, expected = IllegalArgumentException.class)
    public void testPrimsNullGraph() {
        GraphAlgorithms.prims(new Vertex<>('A'), null);
    }

    @Test(timeout = TIMEOUT, expected = IllegalArgumentException.class)
    public void testPrimsNullStart() {
        GraphAlgorithms.prims(null, graph);
    }


    /** ------------------------+   UNDIRECTED GRAPH TESTS   +------------------------
        Graph image: https://imgur.com/a/7JlpQh5
        Adjacency list:
             A	{ B–1, C–7 }
             B	{ A–1, D–6 }
             C	{ A–7, D–10, E–2 F–5 }
             D	{ B–6, C–10, E–3, H–4 }
             E	{ C–2, D–3 }
             F	{ C–5 }
             G	{ }
             H	{ D–4 }
     */

    @Test(timeout = TIMEOUT)
    public void testBFS_A() {
        // Test starting at vertex A
        List<Vertex<Character>> result = GraphAlgorithms.bfs(new Vertex<>('A'), graph);
        List<Vertex<Character>> expected = new LinkedList<>();
        expected.add(new Vertex<>('A'));
        expected.add(new Vertex<>('B'));
        expected.add(new Vertex<>('C'));
        expected.add(new Vertex<>('D'));
        expected.add(new Vertex<>('E'));
        expected.add(new Vertex<>('F'));
        expected.add(new Vertex<>('H'));
        assertEquals(expected, result); // Check amount & order of vertices returned
    }

    @Test(timeout = TIMEOUT)
    public void testBFS_B() {
        // Test starting at vertex B
        List<Vertex<Character>> result = GraphAlgorithms.bfs(new Vertex<>('B'), graph);
        List<Vertex<Character>> expected = new LinkedList<>();
        expected.add(new Vertex<>('B'));
        expected.add(new Vertex<>('A'));
        expected.add(new Vertex<>('D'));
        expected.add(new Vertex<>('C'));
        expected.add(new Vertex<>('E'));
        expected.add(new Vertex<>('H'));
        expected.add(new Vertex<>('F'));
        assertEquals(expected, result); // Check amount & order of vertices returned
    }

    @Test(timeout = TIMEOUT)
    public void testBFS_C() {
        // Test starting at vertex C
        List<Vertex<Character>> result = GraphAlgorithms.bfs(new Vertex<>('C'), graph);
        List<Vertex<Character>> expected = new LinkedList<>();
        expected.add(new Vertex<>('C'));
        expected.add(new Vertex<>('A'));
        expected.add(new Vertex<>('D'));
        expected.add(new Vertex<>('E'));
        expected.add(new Vertex<>('F'));
        expected.add(new Vertex<>('B'));
        expected.add(new Vertex<>('H'));
        assertEquals(expected, result); // Check amount & order of vertices returned
    }

    @Test(timeout = TIMEOUT)
    public void testBFS_D() {
        // Test starting at vertex D
        List<Vertex<Character>> result = GraphAlgorithms.bfs(new Vertex<>('D'), graph);
        List<Vertex<Character>> expected = new LinkedList<>();
        expected.add(new Vertex<>('D'));
        expected.add(new Vertex<>('B'));
        expected.add(new Vertex<>('C'));
        expected.add(new Vertex<>('E'));
        expected.add(new Vertex<>('H'));
        expected.add(new Vertex<>('A'));
        expected.add(new Vertex<>('F'));
        assertEquals(expected, result); // Check amount & order of vertices returned
    }

    @Test(timeout = TIMEOUT)
    public void testBFS_E() {
        // Test starting at vertex E
        List<Vertex<Character>> result = GraphAlgorithms.bfs(new Vertex<>('E'), graph);
        List<Vertex<Character>> expected = new LinkedList<>();
        expected.add(new Vertex<>('E'));
        expected.add(new Vertex<>('C'));
        expected.add(new Vertex<>('D'));
        expected.add(new Vertex<>('A'));
        expected.add(new Vertex<>('F'));
        expected.add(new Vertex<>('B'));
        expected.add(new Vertex<>('H'));
        assertEquals(expected, result); // Check amount & order of vertices returned
    }

    @Test(timeout = TIMEOUT)
    public void testBFS_F() {
        // Test starting at vertex F
        List<Vertex<Character>> result = GraphAlgorithms.bfs(new Vertex<>('F'), graph);
        List<Vertex<Character>> expected = new LinkedList<>();
        expected.add(new Vertex<>('F'));
        expected.add(new Vertex<>('C'));
        expected.add(new Vertex<>('A'));
        expected.add(new Vertex<>('D'));
        expected.add(new Vertex<>('E'));
        expected.add(new Vertex<>('B'));
        expected.add(new Vertex<>('H'));
        assertEquals(expected, result); // Check amount & order of vertices returned
    }

    @Test(timeout = TIMEOUT)
    public void testBFS_G() {
        // Test starting at vertex G
        List<Vertex<Character>> result = GraphAlgorithms.bfs(new Vertex<>('G'), graph);
        List<Vertex<Character>> expected = new LinkedList<>();
        expected.add(new Vertex<>('G'));
        assertEquals(expected, result); // Check amount & order of vertices returned
    }

    @Test(timeout = TIMEOUT)
    public void testBFS_H() {
        // Test starting at vertex H
        List<Vertex<Character>> result = GraphAlgorithms.bfs(new Vertex<>('H'), graph);
        List<Vertex<Character>> expected = new LinkedList<>();
        expected.add(new Vertex<>('H'));
        expected.add(new Vertex<>('D'));
        expected.add(new Vertex<>('B'));
        expected.add(new Vertex<>('C'));
        expected.add(new Vertex<>('E'));
        expected.add(new Vertex<>('A'));
        expected.add(new Vertex<>('F'));
        assertEquals(expected, result); // Check amount & order of vertices returned
    }

    @Test(timeout = TIMEOUT)
    public void testDFS_A() {
        // Test starting at vertex A
        List<Vertex<Character>> result = GraphAlgorithms.dfs(new Vertex<>('A'), graph);
        List<Vertex<Character>> expected = new LinkedList<>();
        expected.add(new Vertex<>('A'));
        expected.add(new Vertex<>('B'));
        expected.add(new Vertex<>('D'));
        expected.add(new Vertex<>('C'));
        expected.add(new Vertex<>('E'));
        expected.add(new Vertex<>('F'));
        expected.add(new Vertex<>('H'));
        assertEquals(expected, result); // Check amount & order of vertices returned
    }

    @Test(timeout = TIMEOUT)
    public void testDFS_B() {
        // Test starting at vertex B
        List<Vertex<Character>> result = GraphAlgorithms.dfs(new Vertex<>('B'), graph);
        List<Vertex<Character>> expected = new LinkedList<>();
        expected.add(new Vertex<>('B'));
        expected.add(new Vertex<>('A'));
        expected.add(new Vertex<>('C'));
        expected.add(new Vertex<>('D'));
        expected.add(new Vertex<>('E'));
        expected.add(new Vertex<>('H'));
        expected.add(new Vertex<>('F'));
        assertEquals(expected, result); // Check amount & order of vertices returned
    }

    @Test(timeout = TIMEOUT)
    public void testDFS_C() {
        // Test starting at vertex C
        List<Vertex<Character>> result = GraphAlgorithms.dfs(new Vertex<>('C'), graph);
        List<Vertex<Character>> expected = new LinkedList<>();
        expected.add(new Vertex<>('C'));
        expected.add(new Vertex<>('A'));
        expected.add(new Vertex<>('B'));
        expected.add(new Vertex<>('D'));
        expected.add(new Vertex<>('E'));
        expected.add(new Vertex<>('H'));
        expected.add(new Vertex<>('F'));
        assertEquals(expected, result); // Check amount & order of vertices returned
    }

    @Test(timeout = TIMEOUT)
    public void testDFS_D() {
        // Test starting at vertex D
        List<Vertex<Character>> result = GraphAlgorithms.dfs(new Vertex<>('D'), graph);
        List<Vertex<Character>> expected = new LinkedList<>();
        expected.add(new Vertex<>('D'));
        expected.add(new Vertex<>('B'));
        expected.add(new Vertex<>('A'));
        expected.add(new Vertex<>('C'));
        expected.add(new Vertex<>('E'));
        expected.add(new Vertex<>('F'));
        expected.add(new Vertex<>('H'));
        assertEquals(expected, result); // Check amount & order of vertices returned
    }
    @Test(timeout = TIMEOUT)
    public void testDFS_E() {
        // Test starting at vertex E
        List<Vertex<Character>> result = GraphAlgorithms.dfs(new Vertex<>('E'), graph);
        List<Vertex<Character>> expected = new LinkedList<>();
        expected.add(new Vertex<>('E'));
        expected.add(new Vertex<>('C'));
        expected.add(new Vertex<>('A'));
        expected.add(new Vertex<>('B'));
        expected.add(new Vertex<>('D'));
        expected.add(new Vertex<>('H'));
        expected.add(new Vertex<>('F'));
        assertEquals(expected, result); // Check amount & order of vertices returned
    }

    @Test(timeout = TIMEOUT)
    public void testDFS_F() {
        // Test starting at vertex F
        List<Vertex<Character>> result = GraphAlgorithms.dfs(new Vertex<>('F'), graph);
        List<Vertex<Character>> expected = new LinkedList<>();
        expected.add(new Vertex<>('F'));
        expected.add(new Vertex<>('C'));
        expected.add(new Vertex<>('A'));
        expected.add(new Vertex<>('B'));
        expected.add(new Vertex<>('D'));
        expected.add(new Vertex<>('E'));
        expected.add(new Vertex<>('H'));
        assertEquals(expected, result); // Check amount & order of vertices returned
    }

    @Test(timeout = TIMEOUT)
    public void testDFS_G() {
        // Test starting at vertex G
        List<Vertex<Character>> result = GraphAlgorithms.dfs(new Vertex<>('G'), graph);
        List<Vertex<Character>> expected = new LinkedList<>();
        expected.add(new Vertex<>('G'));
        assertEquals(expected, result); // Check amount & order of vertices returned
    }

    @Test(timeout = TIMEOUT)
    public void testDFS_H() {
        // Test starting at vertex H
        List<Vertex<Character>> result = GraphAlgorithms.dfs(new Vertex<>('H'), graph);
        List<Vertex<Character>> expected = new LinkedList<>();
        expected.add(new Vertex<>('H'));
        expected.add(new Vertex<>('D'));
        expected.add(new Vertex<>('B'));
        expected.add(new Vertex<>('A'));
        expected.add(new Vertex<>('C'));
        expected.add(new Vertex<>('E'));
        expected.add(new Vertex<>('F'));
        assertEquals(expected, result); // Check amount & order of vertices returned
    }

    @Test(timeout = TIMEOUT)
    public void testDijkstras_A() {
        // Test starting at vertex A
        Map<Vertex<Character>, Integer> result = GraphAlgorithms.dijkstras(new Vertex<>('A'), graph);
        Map<Vertex<Character>, Integer> expected = new HashMap<>();
        expected.put(new Vertex<>('A'), 0);
        expected.put(new Vertex<>('B'), 1);
        expected.put(new Vertex<>('C'), 7);
        expected.put(new Vertex<>('D'), 7);
        expected.put(new Vertex<>('E'), 9);
        expected.put(new Vertex<>('F'), 12);
        expected.put(new Vertex<>('G'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('H'), 11);
        assertEquals(expected, result); // Check amount & order of vertices returned
    }

    @Test(timeout = TIMEOUT)
    public void testDijkstras_B() {
        // Test starting at vertex B
        Map<Vertex<Character>, Integer> result = GraphAlgorithms.dijkstras(new Vertex<>('B'), graph);
        Map<Vertex<Character>, Integer> expected = new HashMap<>();
        expected.put(new Vertex<>('A'), 1);
        expected.put(new Vertex<>('B'), 0);
        expected.put(new Vertex<>('C'), 8);
        expected.put(new Vertex<>('D'), 6);
        expected.put(new Vertex<>('E'), 9);
        expected.put(new Vertex<>('F'), 13);
        expected.put(new Vertex<>('G'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('H'), 10);
        assertEquals(expected, result); // Check amount & order of vertices returned
    }


    @Test(timeout = TIMEOUT)
    public void testDijkstras_C() {
        // Test starting at vertex C
        Map<Vertex<Character>, Integer> result = GraphAlgorithms.dijkstras(new Vertex<>('C'), graph);
        Map<Vertex<Character>, Integer> expected = new HashMap<>();
        expected.put(new Vertex<>('A'), 7);
        expected.put(new Vertex<>('B'), 8);
        expected.put(new Vertex<>('C'), 0);
        expected.put(new Vertex<>('D'), 5);
        expected.put(new Vertex<>('E'), 2);
        expected.put(new Vertex<>('F'), 5);
        expected.put(new Vertex<>('G'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('H'), 9);
        assertEquals(expected, result); // Check amount & order of vertices returned
    }

    @Test(timeout = TIMEOUT)
    public void testDijkstras_D() {
        // Test starting at vertex D
        Map<Vertex<Character>, Integer> result = GraphAlgorithms.dijkstras(new Vertex<>('D'), graph);
        Map<Vertex<Character>, Integer> expected = new HashMap<>();
        expected.put(new Vertex<>('A'), 7);
        expected.put(new Vertex<>('B'), 6);
        expected.put(new Vertex<>('C'), 5);
        expected.put(new Vertex<>('D'), 0);
        expected.put(new Vertex<>('E'), 3);
        expected.put(new Vertex<>('F'), 10);
        expected.put(new Vertex<>('G'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('H'), 4);
        assertEquals(expected, result); // Check amount & order of vertices returned
    }

    @Test(timeout = TIMEOUT)
    public void testDijkstras_E() {
        // Test starting at vertex E
        Map<Vertex<Character>, Integer> result = GraphAlgorithms.dijkstras(new Vertex<>('E'), graph);
        Map<Vertex<Character>, Integer> expected = new HashMap<>();
        expected.put(new Vertex<>('A'), 9);
        expected.put(new Vertex<>('B'), 9);
        expected.put(new Vertex<>('C'), 2);
        expected.put(new Vertex<>('D'), 3);
        expected.put(new Vertex<>('E'), 0);
        expected.put(new Vertex<>('F'), 7);
        expected.put(new Vertex<>('G'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('H'), 7);
        assertEquals(expected, result); // Check amount & order of vertices returned
    }

    @Test(timeout = TIMEOUT)
    public void testDijkstras_F() {
        // Test starting at vertex F
        Map<Vertex<Character>, Integer> result = GraphAlgorithms.dijkstras(new Vertex<>('F'), graph);
        Map<Vertex<Character>, Integer> expected = new HashMap<>();
        expected.put(new Vertex<>('A'), 12);
        expected.put(new Vertex<>('B'), 13);
        expected.put(new Vertex<>('C'), 5);
        expected.put(new Vertex<>('D'), 10);
        expected.put(new Vertex<>('E'), 7);
        expected.put(new Vertex<>('F'), 0);
        expected.put(new Vertex<>('G'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('H'), 14);
        assertEquals(expected, result); // Check amount & order of vertices returned
    }

    @Test(timeout = TIMEOUT)
    public void testDijkstras_G() {
        // Test starting at vertex G
        Map<Vertex<Character>, Integer> result = GraphAlgorithms.dijkstras(new Vertex<>('G'), graph);
        Map<Vertex<Character>, Integer> expected = new HashMap<>();
        expected.put(new Vertex<>('A'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('B'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('C'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('D'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('E'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('F'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('G'), 0);
        expected.put(new Vertex<>('H'), Integer.MAX_VALUE);
        assertEquals(expected, result); // Check amount & order of vertices returned
    }

    @Test(timeout = TIMEOUT)
    public void testDijkstras_H() {
        // Test starting at vertex H
        Map<Vertex<Character>, Integer> result = GraphAlgorithms.dijkstras(new Vertex<>('H'), graph);
        Map<Vertex<Character>, Integer> expected = new HashMap<>();
        expected.put(new Vertex<>('A'), 11);
        expected.put(new Vertex<>('B'), 10);
        expected.put(new Vertex<>('C'), 9);
        expected.put(new Vertex<>('D'), 4);
        expected.put(new Vertex<>('E'), 7);
        expected.put(new Vertex<>('F'), 14);
        expected.put(new Vertex<>('G'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('H'), 0);
        assertEquals(expected, result); // Check amount & order of vertices returned
    }

    private Set<Edge<Character>> primsExpected() {
        Set<Edge<Character>> expected = new HashSet<>();
        expected.add(new Edge<>(new Vertex('A'), new Vertex('B'), 1));
        expected.add(new Edge<>(new Vertex('B'), new Vertex('A'), 1));
        expected.add(new Edge<>(new Vertex('B'), new Vertex('D'), 6));
        expected.add(new Edge<>(new Vertex('D'), new Vertex('B'), 6));
        expected.add(new Edge<>(new Vertex('D'), new Vertex('G'), 4));
        expected.add(new Edge<>(new Vertex('G'), new Vertex('D'), 4));
        expected.add(new Edge<>(new Vertex('D'), new Vertex('E'), 3));
        expected.add(new Edge<>(new Vertex('E'), new Vertex('D'), 3));
        expected.add(new Edge<>(new Vertex('E'), new Vertex('C'), 2));
        expected.add(new Edge<>(new Vertex('C'), new Vertex('E'), 2));
        expected.add(new Edge<>(new Vertex('F'), new Vertex('C'), 5));
        expected.add(new Edge<>(new Vertex('C'), new Vertex('F'), 5));
        return expected;
    }

    /** ------------------------+   CONNECTED GRAPH TESTS (Prim's)   +------------------------
     Graph image: https://imgur.com/a/qov6Qsy
     Adjacency list:
     A	{ B–1, C–7 }
     B	{ A–1, D–6 }
     C	{ A–7, D–10, E–2 F–5 }
     D	{ B–6, C–10, E–3, G–4 }
     E	{ C–2, D–3 }
     F	{ C–5 }
     G	{ D–4 }
     */

    @Test(timeout = TIMEOUT)
    public void testPrimsUnconnectedGraph_1() {
        // Should return null if graph is not fully connected.
        // It is impossible to have a minimum spanning tree if a vertex is inaccessible.
        Set<Edge<Character>> result;
        result = GraphAlgorithms.prims(new Vertex<>('A'), graph);
        assertNull(result);
        result = GraphAlgorithms.prims(new Vertex<>('B'), graph);
        assertNull(result);
        result = GraphAlgorithms.prims(new Vertex<>('C'), graph);
        assertNull(result);
        result = GraphAlgorithms.prims(new Vertex<>('D'), graph);
        assertNull(result);
        result = GraphAlgorithms.prims(new Vertex<>('E'), graph);
        assertNull(result);
        result = GraphAlgorithms.prims(new Vertex<>('F'), graph);
        assertNull(result);
        result = GraphAlgorithms.prims(new Vertex<>('G'), graph);
        assertNull(result);
        result = GraphAlgorithms.prims(new Vertex<>('H'), graph);
        assertNull(result);
    }

    @Test(timeout = TIMEOUT)
    public void testPrimsUnconnectedLargeGraph() {
        // Should return null if graph is not fully connected.
        // It is impossible to have a minimum spanning tree if a vertex is inaccessible.
        Set<Edge<Character>> result;
        result = GraphAlgorithms.prims(new Vertex<>('A'), largeGraph);
        assertNull(result);
        result = GraphAlgorithms.prims(new Vertex<>('B'), largeGraph);
        assertNull(result);
        result = GraphAlgorithms.prims(new Vertex<>('C'), largeGraph);
        assertNull(result);
        result = GraphAlgorithms.prims(new Vertex<>('D'), largeGraph);
        assertNull(result);
        result = GraphAlgorithms.prims(new Vertex<>('E'), largeGraph);
        assertNull(result);
        result = GraphAlgorithms.prims(new Vertex<>('F'), largeGraph);
        assertNull(result);
        result = GraphAlgorithms.prims(new Vertex<>('G'), largeGraph);
        assertNull(result);
        result = GraphAlgorithms.prims(new Vertex<>('H'), largeGraph);
        assertNull(result);
        result = GraphAlgorithms.prims(new Vertex<>('I'), largeGraph);
        assertNull(result);
        result = GraphAlgorithms.prims(new Vertex<>('J'), largeGraph);
        assertNull(result);
        result = GraphAlgorithms.prims(new Vertex<>('K'), largeGraph);
        assertNull(result);
        result = GraphAlgorithms.prims(new Vertex<>('L'), largeGraph);
        assertNull(result);
        result = GraphAlgorithms.prims(new Vertex<>('M'), largeGraph);
        assertNull(result);
        result = GraphAlgorithms.prims(new Vertex<>('N'), largeGraph);
        assertNull(result);
        result = GraphAlgorithms.prims(new Vertex<>('O'), largeGraph);
        assertNull(result);
        result = GraphAlgorithms.prims(new Vertex<>('P'), largeGraph);
        assertNull(result);
        result = GraphAlgorithms.prims(new Vertex<>('Q'), largeGraph);
        assertNull(result);
        result = GraphAlgorithms.prims(new Vertex<>('R'), largeGraph);
        assertNull(result);
        result = GraphAlgorithms.prims(new Vertex<>('S'), largeGraph);
        assertNull(result);
        result = GraphAlgorithms.prims(new Vertex<>('T'), largeGraph);
        assertNull(result);
        result = GraphAlgorithms.prims(new Vertex<>('U'), largeGraph);
        assertNull(result);
        result = GraphAlgorithms.prims(new Vertex<>('V'), largeGraph);
        assertNull(result);
        result = GraphAlgorithms.prims(new Vertex<>('W'), largeGraph);
        assertNull(result);
        result = GraphAlgorithms.prims(new Vertex<>('X'), largeGraph);
        assertNull(result);
        result = GraphAlgorithms.prims(new Vertex<>('Y'), largeGraph);
        assertNull(result);
        result = GraphAlgorithms.prims(new Vertex<>('Z'), largeGraph);
        assertNull(result);
    }

    @Test(timeout = TIMEOUT)
    public void testPrims_A() {
        // Test starting at vertex A
        Set<Edge<Character>> result = GraphAlgorithms.prims(new Vertex<>('A'), connectedGraph);
        assertEquals(primsExpected(), result); // Check amount & order of vertices returned
    }

    @Test(timeout = TIMEOUT)
    public void testPrims_B() {
        // Test starting at vertex B
        Set<Edge<Character>> result = GraphAlgorithms.prims(new Vertex<>('B'), connectedGraph);
        assertEquals(primsExpected(), result); // Check amount & order of vertices returned
    }

    @Test(timeout = TIMEOUT)
    public void testPrims_C() {
        // Test starting at vertex C
        Set<Edge<Character>> result = GraphAlgorithms.prims(new Vertex<>('C'), connectedGraph);
        assertEquals(primsExpected(), result); // Check amount & order of vertices returned
    }

    @Test(timeout = TIMEOUT)
    public void testPrims_D() {
        // Test starting at vertex D
        Set<Edge<Character>> result = GraphAlgorithms.prims(new Vertex<>('D'), connectedGraph);
        assertEquals(primsExpected(), result); // Check amount & order of vertices returned
    }

    @Test(timeout = TIMEOUT)
    public void testPrims_E() {
        // Test starting at vertex E
        Set<Edge<Character>> result = GraphAlgorithms.prims(new Vertex<>('E'), connectedGraph);
        assertEquals(primsExpected(), result); // Check amount & order of vertices returned
    }

    @Test(timeout = TIMEOUT)
    public void testPrims_F() {
        // Test starting at vertex F
        Set<Edge<Character>> result = GraphAlgorithms.prims(new Vertex<>('F'), connectedGraph);
        assertEquals(primsExpected(), result); // Check amount & order of vertices returned
    }

    @Test(timeout = TIMEOUT)
    public void testPrims_G() {
        // Test starting at vertex G
        Set<Edge<Character>> result = GraphAlgorithms.prims(new Vertex<>('G'), connectedGraph);
        assertEquals(primsExpected(), result); // Check amount & order of vertices returned
    }


    /** ------------------------+   DIRECTED GRAPH TESTS   +------------------------
        Graph image: https://imgur.com/a/oNOygQ9
        Adjacency list:
             A	{ G–8 }
             B	{  }
             C	{ F–4 }
             D	{ G–2 }
             E	{ H–0 }
             F	{ G–10 }
             G	{ C–6, D–2, H–3 }
             H	{ A–4, E–0 }
     */

    @Test(timeout = TIMEOUT)
    public void testBFSDirected_A() {
        // Test starting at vertex A
        List<Vertex<Character>> result = GraphAlgorithms.bfs(new Vertex<>('A'), directedGraph);
        List<Vertex<Character>> expected = new LinkedList<>();
        expected.add(new Vertex<>('A'));
        expected.add(new Vertex<>('G'));
        expected.add(new Vertex<>('C'));
        expected.add(new Vertex<>('D'));
        expected.add(new Vertex<>('H'));
        expected.add(new Vertex<>('F'));
        expected.add(new Vertex<>('E'));
        assertEquals(expected, result); // Check amount & order of vertices returned
    }

    @Test(timeout = TIMEOUT)
    public void testBFSDirected_B() {
        // Test starting at vertex B
        List<Vertex<Character>> result = GraphAlgorithms.bfs(new Vertex<>('B'), directedGraph);
        List<Vertex<Character>> expected = new LinkedList<>();
        expected.add(new Vertex<>('B'));
        assertEquals(expected, result); // Check amount & order of vertices returned
    }

    @Test(timeout = TIMEOUT)
    public void testBFSDirected_C() {
        // Test starting at vertex C
        List<Vertex<Character>> result = GraphAlgorithms.bfs(new Vertex<>('C'), directedGraph);
        List<Vertex<Character>> expected = new LinkedList<>();
        expected.add(new Vertex<>('C'));
        expected.add(new Vertex<>('F'));
        expected.add(new Vertex<>('G'));
        expected.add(new Vertex<>('D'));
        expected.add(new Vertex<>('H'));
        expected.add(new Vertex<>('A'));
        expected.add(new Vertex<>('E'));
        assertEquals(expected, result); // Check amount & order of vertices returned
    }

    @Test(timeout = TIMEOUT)
    public void testBFSDirected_D() {
        // Test starting at vertex D
        List<Vertex<Character>> result = GraphAlgorithms.bfs(new Vertex<>('D'), directedGraph);
        List<Vertex<Character>> expected = new LinkedList<>();
        expected.add(new Vertex<>('D'));
        expected.add(new Vertex<>('G'));
        expected.add(new Vertex<>('C'));
        expected.add(new Vertex<>('H'));
        expected.add(new Vertex<>('F'));
        expected.add(new Vertex<>('A'));
        expected.add(new Vertex<>('E'));
        assertEquals(expected, result); // Check amount & order of vertices returned
    }

    @Test(timeout = TIMEOUT)
    public void testBFSDirected_E() {
        // Test starting at vertex E
        List<Vertex<Character>> result = GraphAlgorithms.bfs(new Vertex<>('E'), directedGraph);
        List<Vertex<Character>> expected = new LinkedList<>();
        expected.add(new Vertex<>('E'));
        expected.add(new Vertex<>('H'));
        expected.add(new Vertex<>('A'));
        expected.add(new Vertex<>('G'));
        expected.add(new Vertex<>('C'));
        expected.add(new Vertex<>('D'));
        expected.add(new Vertex<>('F'));
        assertEquals(expected, result); // Check amount & order of vertices returned
    }

    @Test(timeout = TIMEOUT)
    public void testBFSDirected_F() {
        // Test starting at vertex F
        List<Vertex<Character>> result = GraphAlgorithms.bfs(new Vertex<>('F'), directedGraph);
        List<Vertex<Character>> expected = new LinkedList<>();
        expected.add(new Vertex<>('F'));
        expected.add(new Vertex<>('G'));
        expected.add(new Vertex<>('C'));
        expected.add(new Vertex<>('D'));
        expected.add(new Vertex<>('H'));
        expected.add(new Vertex<>('A'));
        expected.add(new Vertex<>('E'));
        assertEquals(expected, result); // Check amount & order of vertices returned
    }

    @Test(timeout = TIMEOUT)
    public void testBFSDirected_G() {
        // Test starting at vertex G
        List<Vertex<Character>> result = GraphAlgorithms.bfs(new Vertex<>('G'), directedGraph);
        List<Vertex<Character>> expected = new LinkedList<>();
        expected.add(new Vertex<>('G'));
        expected.add(new Vertex<>('C'));
        expected.add(new Vertex<>('D'));
        expected.add(new Vertex<>('H'));
        expected.add(new Vertex<>('F'));
        expected.add(new Vertex<>('A'));
        expected.add(new Vertex<>('E'));
        assertEquals(expected, result); // Check amount & order of vertices returned
    }

    @Test(timeout = TIMEOUT)
    public void testBFSDirected_H() {
        // Test starting at vertex H
        List<Vertex<Character>> result = GraphAlgorithms.bfs(new Vertex<>('H'), directedGraph);
        List<Vertex<Character>> expected = new LinkedList<>();
        expected.add(new Vertex<>('H'));
        expected.add(new Vertex<>('A'));
        expected.add(new Vertex<>('E'));
        expected.add(new Vertex<>('G'));
        expected.add(new Vertex<>('C'));
        expected.add(new Vertex<>('D'));
        expected.add(new Vertex<>('F'));
        assertEquals(expected, result); // Check amount & order of vertices returned
    }

    @Test(timeout = TIMEOUT)
    public void testDFSDirected_A() {
        // Test starting at vertex A
        List<Vertex<Character>> result = GraphAlgorithms.dfs(new Vertex<>('A'), directedGraph);
        List<Vertex<Character>> expected = new LinkedList<>();
        expected.add(new Vertex<>('A'));
        expected.add(new Vertex<>('G'));
        expected.add(new Vertex<>('C'));
        expected.add(new Vertex<>('F'));
        expected.add(new Vertex<>('D'));
        expected.add(new Vertex<>('H'));
        expected.add(new Vertex<>('E'));
        assertEquals(expected, result); // Check amount & order of vertices returned
    }

    @Test(timeout = TIMEOUT)
    public void testDFSDirected_B() {
        // Test starting at vertex B
        List<Vertex<Character>> result = GraphAlgorithms.dfs(new Vertex<>('B'), directedGraph);
        List<Vertex<Character>> expected = new LinkedList<>();
        expected.add(new Vertex<>('B'));
        assertEquals(expected, result); // Check amount & order of vertices returned
    }

    @Test(timeout = TIMEOUT)
    public void testDFSDirected_C() {
        // Test starting at vertex C
        List<Vertex<Character>> result = GraphAlgorithms.dfs(new Vertex<>('C'), directedGraph);
        List<Vertex<Character>> expected = new LinkedList<>();
        expected.add(new Vertex<>('C'));
        expected.add(new Vertex<>('F'));
        expected.add(new Vertex<>('G'));
        expected.add(new Vertex<>('D'));
        expected.add(new Vertex<>('H'));
        expected.add(new Vertex<>('A'));
        expected.add(new Vertex<>('E'));
        assertEquals(expected, result); // Check amount & order of vertices returned
    }

    @Test(timeout = TIMEOUT)
    public void testDFSDirected_D() {
        // Test starting at vertex D
        List<Vertex<Character>> result = GraphAlgorithms.dfs(new Vertex<>('D'), directedGraph);
        List<Vertex<Character>> expected = new LinkedList<>();
        expected.add(new Vertex<>('D'));
        expected.add(new Vertex<>('G'));
        expected.add(new Vertex<>('C'));
        expected.add(new Vertex<>('F'));
        expected.add(new Vertex<>('H'));
        expected.add(new Vertex<>('A'));
        expected.add(new Vertex<>('E'));
        assertEquals(expected, result); // Check amount & order of vertices returned
    }

    @Test(timeout = TIMEOUT)
    public void testDFSDirected_E() {
        // Test starting at vertex E
        List<Vertex<Character>> result = GraphAlgorithms.dfs(new Vertex<>('E'), directedGraph);
        List<Vertex<Character>> expected = new LinkedList<>();
        expected.add(new Vertex<>('E'));
        expected.add(new Vertex<>('H'));
        expected.add(new Vertex<>('A'));
        expected.add(new Vertex<>('G'));
        expected.add(new Vertex<>('C'));
        expected.add(new Vertex<>('F'));
        expected.add(new Vertex<>('D'));
        assertEquals(expected, result); // Check amount & order of vertices returned
    }

    @Test(timeout = TIMEOUT)
    public void testDFSDirected_F() {
        // Test starting at vertex F
        List<Vertex<Character>> result = GraphAlgorithms.dfs(new Vertex<>('F'), directedGraph);
        List<Vertex<Character>> expected = new LinkedList<>();
        expected.add(new Vertex<>('F'));
        expected.add(new Vertex<>('G'));
        expected.add(new Vertex<>('C'));
        expected.add(new Vertex<>('D'));
        expected.add(new Vertex<>('H'));
        expected.add(new Vertex<>('A'));
        expected.add(new Vertex<>('E'));
        assertEquals(expected, result); // Check amount & order of vertices returned
    }

    @Test(timeout = TIMEOUT)
    public void testDFSDirected_G() {
        // Test starting at vertex G
        List<Vertex<Character>> result = GraphAlgorithms.dfs(new Vertex<>('G'), directedGraph);
        List<Vertex<Character>> expected = new LinkedList<>();
        expected.add(new Vertex<>('G'));
        expected.add(new Vertex<>('C'));
        expected.add(new Vertex<>('F'));
        expected.add(new Vertex<>('D'));
        expected.add(new Vertex<>('H'));
        expected.add(new Vertex<>('A'));
        expected.add(new Vertex<>('E'));
        assertEquals(expected, result); // Check amount & order of vertices returned
    }

    @Test(timeout = TIMEOUT)
    public void testDFSDirected_H() {
        // Test starting at vertex H
        List<Vertex<Character>> result = GraphAlgorithms.dfs(new Vertex<>('H'), directedGraph);
        List<Vertex<Character>> expected = new LinkedList<>();
        expected.add(new Vertex<>('H'));
        expected.add(new Vertex<>('A'));
        expected.add(new Vertex<>('G'));
        expected.add(new Vertex<>('C'));
        expected.add(new Vertex<>('F'));
        expected.add(new Vertex<>('D'));
        expected.add(new Vertex<>('E'));
        assertEquals(expected, result); // Check amount & order of vertices returned
    }

    @Test(timeout = TIMEOUT)
    public void testDijkstrasDirected_A() {
        // Test starting at vertex A
        Map<Vertex<Character>, Integer> result = GraphAlgorithms.dijkstras(new Vertex<>('A'), directedGraph);
        Map<Vertex<Character>, Integer> expected = new HashMap<>();
        expected.put(new Vertex<>('A'), 0);
        expected.put(new Vertex<>('B'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('C'), 14);
        expected.put(new Vertex<>('D'), 10);
        expected.put(new Vertex<>('E'), 11);
        expected.put(new Vertex<>('F'), 18);
        expected.put(new Vertex<>('G'), 8);
        expected.put(new Vertex<>('H'), 11);
        assertEquals(expected, result); // Check amount & order of vertices returned
    }

    @Test(timeout = TIMEOUT)
    public void testDijkstrasDirected_B() {
        // Test starting at vertex B
        Map<Vertex<Character>, Integer> result = GraphAlgorithms.dijkstras(new Vertex<>('B'), directedGraph);
        Map<Vertex<Character>, Integer> expected = new HashMap<>();
        expected.put(new Vertex<>('A'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('B'), 0);
        expected.put(new Vertex<>('C'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('D'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('E'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('F'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('G'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('H'), Integer.MAX_VALUE);
        assertEquals(expected, result); // Check amount & order of vertices returned
    }

    @Test(timeout = TIMEOUT)
    public void testDijkstrasDirected_C() {
        // Test starting at vertex C
        Map<Vertex<Character>, Integer> result = GraphAlgorithms.dijkstras(new Vertex<>('C'), directedGraph);
        Map<Vertex<Character>, Integer> expected = new HashMap<>();
        expected.put(new Vertex<>('A'), 21);
        expected.put(new Vertex<>('B'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('C'), 0);
        expected.put(new Vertex<>('D'), 16);
        expected.put(new Vertex<>('E'), 17);
        expected.put(new Vertex<>('F'), 4);
        expected.put(new Vertex<>('G'), 14);
        expected.put(new Vertex<>('H'), 17);
        assertEquals(expected, result); // Check amount & order of vertices returned
    }

    @Test(timeout = TIMEOUT)
    public void testDijkstrasDirected_D() {
        // Test starting at vertex D
        Map<Vertex<Character>, Integer> result = GraphAlgorithms.dijkstras(new Vertex<>('D'), directedGraph);
        Map<Vertex<Character>, Integer> expected = new HashMap<>();
        expected.put(new Vertex<>('A'), 9);
        expected.put(new Vertex<>('B'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('C'), 8);
        expected.put(new Vertex<>('D'), 0);
        expected.put(new Vertex<>('E'), 5);
        expected.put(new Vertex<>('F'), 12);
        expected.put(new Vertex<>('G'), 2);
        expected.put(new Vertex<>('H'), 5);
        assertEquals(expected, result); // Check amount & order of vertices returned
    }

    @Test(timeout = TIMEOUT)
    public void testDijkstrasDirected_E() {
        // Test starting at vertex E
        Map<Vertex<Character>, Integer> result = GraphAlgorithms.dijkstras(new Vertex<>('E'), directedGraph);
        Map<Vertex<Character>, Integer> expected = new HashMap<>();
        expected.put(new Vertex<>('A'), 4);
        expected.put(new Vertex<>('B'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('C'), 18);
        expected.put(new Vertex<>('D'), 14);
        expected.put(new Vertex<>('E'), 0);
        expected.put(new Vertex<>('F'), 22);
        expected.put(new Vertex<>('G'), 12);
        expected.put(new Vertex<>('H'), 0);
        assertEquals(expected, result); // Check amount & order of vertices returned
    }

    @Test(timeout = TIMEOUT)
    public void testDijkstrasDirected_F() {
        // Test starting at vertex F
        Map<Vertex<Character>, Integer> result = GraphAlgorithms.dijkstras(new Vertex<>('F'), directedGraph);
        Map<Vertex<Character>, Integer> expected = new HashMap<>();
        expected.put(new Vertex<>('A'), 17);
        expected.put(new Vertex<>('B'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('C'), 16);
        expected.put(new Vertex<>('D'), 12);
        expected.put(new Vertex<>('E'), 13);
        expected.put(new Vertex<>('F'), 0);
        expected.put(new Vertex<>('G'), 10);
        expected.put(new Vertex<>('H'), 13);
        assertEquals(expected, result); // Check amount & order of vertices returned
    }

    @Test(timeout = TIMEOUT)
    public void testDijkstrasDirected_G() {
        // Test starting at vertex G
        Map<Vertex<Character>, Integer> result = GraphAlgorithms.dijkstras(new Vertex<>('G'), directedGraph);
        Map<Vertex<Character>, Integer> expected = new HashMap<>();
        expected.put(new Vertex<>('A'), 7);
        expected.put(new Vertex<>('B'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('C'), 6);
        expected.put(new Vertex<>('D'), 2);
        expected.put(new Vertex<>('E'), 3);
        expected.put(new Vertex<>('F'), 10);
        expected.put(new Vertex<>('G'), 0);
        expected.put(new Vertex<>('H'), 3);
        assertEquals(expected, result); // Check amount & order of vertices returned
    }

    @Test(timeout = TIMEOUT)
    public void testDijkstrasDirected_H() {
        // Test starting at vertex H
        Map<Vertex<Character>, Integer> result = GraphAlgorithms.dijkstras(new Vertex<>('H'), directedGraph);
        Map<Vertex<Character>, Integer> expected = new HashMap<>();
        expected.put(new Vertex<>('A'), 4);
        expected.put(new Vertex<>('B'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('C'), 18);
        expected.put(new Vertex<>('D'), 14);
        expected.put(new Vertex<>('E'), 0);
        expected.put(new Vertex<>('F'), 22);
        expected.put(new Vertex<>('G'), 12);
        expected.put(new Vertex<>('H'), 0);
        assertEquals(expected, result); // Check amount & order of vertices returned
    }

    /** ------------------------+  LARGE CONNECTED GRAPH TESTS (Prim's)   +------------------------
         Graph image: https://imgur.com/a/XaeXISY
         Adjacency list:
             A	{ B–0, C–2, J–7, V–3 }
             B	{ A–0, Y–1, Z–4 }
             C	{ A–2, L–0, Z–3 }
             G	{ T–3, W–3 }
             J	{ A–7, N–6, U–1 }
             L	{ C–0 }
             N	{ J–6, S–7, Y–9 }
             P	{ T–2, U–4, W–6 }
             S	{ N–7 }
             T	{ G–3, P–2 }
             U	{ J–1, P–4 }
             V	{ A–3 }
             W	{ G–3, P–6 }
             Y	{ B–1, N–9, Z–8 }
             Z	{ B–4, C–3, Y–8 }
     */

    private Set<Edge<Character>> primsExpectedLarge() {
        Set<Edge<Character>> expected = new HashSet<>();
        expected.add(new Edge<>(new Vertex('A'), new Vertex('C'), 2));
        expected.add(new Edge<>(new Vertex('C'), new Vertex('A'), 2));
        expected.add(new Edge<>(new Vertex('U'), new Vertex('P'), 4));
        expected.add(new Edge<>(new Vertex('P'), new Vertex('U'), 4));
        expected.add(new Edge<>(new Vertex('J'), new Vertex('N'), 6));
        expected.add(new Edge<>(new Vertex('N'), new Vertex('J'), 6));
        expected.add(new Edge<>(new Vertex('A'), new Vertex('B'), 0));
        expected.add(new Edge<>(new Vertex('B'), new Vertex('A'), 0));
        expected.add(new Edge<>(new Vertex('P'), new Vertex('T'), 2));
        expected.add(new Edge<>(new Vertex('T'), new Vertex('P'), 2));
        expected.add(new Edge<>(new Vertex('A'), new Vertex('J'), 7));
        expected.add(new Edge<>(new Vertex('J'), new Vertex('A'), 7));
        expected.add(new Edge<>(new Vertex('C'), new Vertex('L'), 0));
        expected.add(new Edge<>(new Vertex('L'), new Vertex('C'), 0));
        expected.add(new Edge<>(new Vertex('T'), new Vertex('G'), 3));
        expected.add(new Edge<>(new Vertex('G'), new Vertex('T'), 3));
        expected.add(new Edge<>(new Vertex('G'), new Vertex('W'), 3));
        expected.add(new Edge<>(new Vertex('W'), new Vertex('G'), 3));
        expected.add(new Edge<>(new Vertex('A'), new Vertex('V'), 3));
        expected.add(new Edge<>(new Vertex('V'), new Vertex('A'), 3));
        expected.add(new Edge<>(new Vertex('B'), new Vertex('Y'), 1));
        expected.add(new Edge<>(new Vertex('Y'), new Vertex('B'), 1));
        expected.add(new Edge<>(new Vertex('C'), new Vertex('Z'), 3));
        expected.add(new Edge<>(new Vertex('Z'), new Vertex('C'), 3));
        expected.add(new Edge<>(new Vertex('N'), new Vertex('S'), 7));
        expected.add(new Edge<>(new Vertex('S'), new Vertex('N'), 7));
        expected.add(new Edge<>(new Vertex('J'), new Vertex('U'), 1));
        expected.add(new Edge<>(new Vertex('U'), new Vertex('J'), 1));
        return expected;
    }

    @Test(timeout = TIMEOUT)
    public void testPrimsLarge_A() {
        // Test starting at vertex A
        Set<Edge<Character>> result = GraphAlgorithms.prims(new Vertex<>('A'), largeConnectedGraph);
        assertEquals(primsExpectedLarge(), result); // Check amount & order of vertices returned
    }

    @Test(timeout = TIMEOUT)
    public void testPrimsLarge_B() {
        // Test starting at vertex B
        Set<Edge<Character>> result = GraphAlgorithms.prims(new Vertex<>('B'), largeConnectedGraph);
        assertEquals(primsExpectedLarge(), result); // Check amount & order of vertices returned
    }

    @Test(timeout = TIMEOUT)
    public void testPrimsLarge_C() {
        // Test starting at vertex C
        Set<Edge<Character>> result = GraphAlgorithms.prims(new Vertex<>('C'), largeConnectedGraph);
        assertEquals(primsExpectedLarge(), result); // Check amount & order of vertices returned
    }

    @Test(timeout = TIMEOUT)
    public void testPrimsLarge_G() {
        // Test starting at vertex G
        Set<Edge<Character>> result = GraphAlgorithms.prims(new Vertex<>('G'), largeConnectedGraph);
        assertEquals(primsExpectedLarge(), result); // Check amount & order of vertices returned
    }

    @Test(timeout = TIMEOUT)
    public void testPrimsLarge_J() {
        // Test starting at vertex J
        Set<Edge<Character>> result = GraphAlgorithms.prims(new Vertex<>('J'), largeConnectedGraph);
        assertEquals(primsExpectedLarge(), result); // Check amount & order of vertices returned
    }

    @Test(timeout = TIMEOUT)
    public void testPrimsLarge_L() {
        // Test starting at vertex L
        Set<Edge<Character>> result = GraphAlgorithms.prims(new Vertex<>('L'), largeConnectedGraph);
        assertEquals(primsExpectedLarge(), result); // Check amount & order of vertices returned
    }

    @Test(timeout = TIMEOUT)
    public void testPrimsLarge_N() {
        // Test starting at vertex N
        Set<Edge<Character>> result = GraphAlgorithms.prims(new Vertex<>('N'), largeConnectedGraph);
        assertEquals(primsExpectedLarge(), result); // Check amount & order of vertices returned
    }

    @Test(timeout = TIMEOUT)
    public void testPrimsLarge_P() {
        // Test starting at vertex P
        Set<Edge<Character>> result = GraphAlgorithms.prims(new Vertex<>('P'), largeConnectedGraph);
        assertEquals(primsExpectedLarge(), result); // Check amount & order of vertices returned
    }

    @Test(timeout = TIMEOUT)
    public void testPrimsLarge_S() {
        // Test starting at vertex S
        Set<Edge<Character>> result = GraphAlgorithms.prims(new Vertex<>('S'), largeConnectedGraph);
        assertEquals(primsExpectedLarge(), result); // Check amount & order of vertices returned
    }

    @Test(timeout = TIMEOUT)
    public void testPrimsLarge_T() {
        // Test starting at vertex T
        Set<Edge<Character>> result = GraphAlgorithms.prims(new Vertex<>('T'), largeConnectedGraph);
        assertEquals(primsExpectedLarge(), result); // Check amount & order of vertices returned
    }

    @Test(timeout = TIMEOUT)
    public void testPrimsLarge_U() {
        // Test starting at vertex U
        Set<Edge<Character>> result = GraphAlgorithms.prims(new Vertex<>('U'), largeConnectedGraph);
        assertEquals(primsExpectedLarge(), result); // Check amount & order of vertices returned
    }

    @Test(timeout = TIMEOUT)
    public void testPrimsLarge_V() {
        // Test starting at vertex V
        Set<Edge<Character>> result = GraphAlgorithms.prims(new Vertex<>('V'), largeConnectedGraph);
        assertEquals(primsExpectedLarge(), result); // Check amount & order of vertices returned
    }


    @Test(timeout = TIMEOUT)
    public void testPrimsLarge_W() {
        // Test starting at vertex W
        Set<Edge<Character>> result = GraphAlgorithms.prims(new Vertex<>('W'), largeConnectedGraph);
        assertEquals(primsExpectedLarge(), result); // Check amount & order of vertices returned
    }

    @Test(timeout = TIMEOUT)
    public void testPrimsLarge_Y() {
        // Test starting at vertex Y
        Set<Edge<Character>> result = GraphAlgorithms.prims(new Vertex<>('Y'), largeConnectedGraph);
        assertEquals(primsExpectedLarge(), result); // Check amount & order of vertices returned
    }

    @Test(timeout = TIMEOUT)
    public void testPrimsLarge_Z() {
        // Test starting at vertex Z
        Set<Edge<Character>> result = GraphAlgorithms.prims(new Vertex<>('Z'), largeConnectedGraph);
        assertEquals(primsExpectedLarge(), result); // Check amount & order of vertices returned
    }


    /** ------------------------+   GENERATED TESTS   +------------------------
        Graph image: https://imgur.com/a/DOBL8ug
        Adjacency list:
             A	{ B–0, C–2, J–7, V–3 }
             B	{ A–0, Y–1, Z–4 }
             C	{ A–2, L–0, Z–3 }
             D	{ M–3, O–2 }
             E	{ }
             F	{ M–7, O–1 }
             G	{ T–3, W–3 }
             H	{ M–6 }
             I	{ Q–6, X–7 }
             J	{ A–7, N–6, U–1 }
             K	{ R–9 }
             L	{ C–0 }
             M	{ D–3, F–7, H–6, O–8 }
             N	{ J–6, S–7, Y–9 }
             O	{ D–2, F–1, M–8, Q–4 }
             P	{ T–2, U–4, W–6 }
             Q	{ I–6, O–4, X–5 }
             R	{ K–9 }
             S	{ N–7 }
             T	{ G–3, P–2 }
             U	{ J–1, P–4 }
             V	{ A–3 }
             W	{ G–3, P–6 }
             X	{ I–7, Q–5 }
             Y	{ B–1, N–9, Z–8 }
             Z	{ B–4, C–3, Y–8 }
     */

    @Test(timeout = TIMEOUT)
    public void testDijkstrasLarge_A() {
        // Test starting at vertex A
        Map<Vertex<Character>, Integer> expected = new HashMap<>();
        expected.put(new Vertex<>('A'), 0);
        expected.put(new Vertex<>('B'), 0);
        expected.put(new Vertex<>('C'), 2);
        expected.put(new Vertex<>('D'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('E'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('F'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('G'), 17);
        expected.put(new Vertex<>('H'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('I'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('J'), 7);
        expected.put(new Vertex<>('K'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('L'), 2);
        expected.put(new Vertex<>('M'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('N'), 10);
        expected.put(new Vertex<>('O'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('P'), 12);
        expected.put(new Vertex<>('Q'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('R'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('S'), 17);
        expected.put(new Vertex<>('T'), 14);
        expected.put(new Vertex<>('U'), 8);
        expected.put(new Vertex<>('V'), 3);
        expected.put(new Vertex<>('W'), 18);
        expected.put(new Vertex<>('X'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('Y'), 1);
        expected.put(new Vertex<>('Z'), 4);

        System.out.println("Dijkstras starting from A");
        Map<Vertex<Character>, Integer> result = GraphAlgorithms.dijkstras(new Vertex<>('A'), largeGraph);
        assertEquals(expected, result); // Check amount & order of vertices returned
    }

    @Test(timeout = TIMEOUT)
    public void testDijkstrasLarge_B() {
        // Test starting at vertex B
        Map<Vertex<Character>, Integer> expected = new HashMap<>();
        expected.put(new Vertex<>('A'), 0);
        expected.put(new Vertex<>('B'), 0);
        expected.put(new Vertex<>('C'), 2);
        expected.put(new Vertex<>('D'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('E'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('F'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('G'), 17);
        expected.put(new Vertex<>('H'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('I'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('J'), 7);
        expected.put(new Vertex<>('K'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('L'), 2);
        expected.put(new Vertex<>('M'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('N'), 10);
        expected.put(new Vertex<>('O'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('P'), 12);
        expected.put(new Vertex<>('Q'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('R'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('S'), 17);
        expected.put(new Vertex<>('T'), 14);
        expected.put(new Vertex<>('U'), 8);
        expected.put(new Vertex<>('V'), 3);
        expected.put(new Vertex<>('W'), 18);
        expected.put(new Vertex<>('X'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('Y'), 1);
        expected.put(new Vertex<>('Z'), 4);

        System.out.println("Dijkstras starting from B");
        Map<Vertex<Character>, Integer> result = GraphAlgorithms.dijkstras(new Vertex<>('B'), largeGraph);
        assertEquals(expected, result); // Check amount & order of vertices returned
    }

    @Test(timeout = TIMEOUT)
    public void testDijkstrasLarge_C() {
        // Test starting at vertex C
        Map<Vertex<Character>, Integer> expected = new HashMap<>();
        expected.put(new Vertex<>('A'), 2);
        expected.put(new Vertex<>('B'), 2);
        expected.put(new Vertex<>('C'), 0);
        expected.put(new Vertex<>('D'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('E'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('F'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('G'), 19);
        expected.put(new Vertex<>('H'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('I'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('J'), 9);
        expected.put(new Vertex<>('K'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('L'), 0);
        expected.put(new Vertex<>('M'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('N'), 12);
        expected.put(new Vertex<>('O'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('P'), 14);
        expected.put(new Vertex<>('Q'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('R'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('S'), 19);
        expected.put(new Vertex<>('T'), 16);
        expected.put(new Vertex<>('U'), 10);
        expected.put(new Vertex<>('V'), 5);
        expected.put(new Vertex<>('W'), 20);
        expected.put(new Vertex<>('X'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('Y'), 3);
        expected.put(new Vertex<>('Z'), 3);

        System.out.println("Dijkstras starting from C");
        Map<Vertex<Character>, Integer> result = GraphAlgorithms.dijkstras(new Vertex<>('C'), largeGraph);
        assertEquals(expected, result); // Check amount & order of vertices returned
    }

    @Test(timeout = TIMEOUT)
    public void testDijkstrasLarge_D() {
        // Test starting at vertex D
        Map<Vertex<Character>, Integer> expected = new HashMap<>();
        expected.put(new Vertex<>('A'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('B'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('C'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('D'), 0);
        expected.put(new Vertex<>('E'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('F'), 3);
        expected.put(new Vertex<>('G'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('H'), 9);
        expected.put(new Vertex<>('I'), 12);
        expected.put(new Vertex<>('J'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('K'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('L'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('M'), 3);
        expected.put(new Vertex<>('N'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('O'), 2);
        expected.put(new Vertex<>('P'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('Q'), 6);
        expected.put(new Vertex<>('R'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('S'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('T'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('U'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('V'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('W'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('X'), 11);
        expected.put(new Vertex<>('Y'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('Z'), Integer.MAX_VALUE);

        System.out.println("Dijkstras starting from D");
        Map<Vertex<Character>, Integer> result = GraphAlgorithms.dijkstras(new Vertex<>('D'), largeGraph);
        assertEquals(expected, result); // Check amount & order of vertices returned
    }

    @Test(timeout = TIMEOUT)
    public void testDijkstrasLarge_E() {
        // Test starting at vertex E
        Map<Vertex<Character>, Integer> expected = new HashMap<>();
        expected.put(new Vertex<>('A'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('B'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('C'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('D'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('E'), 0);
        expected.put(new Vertex<>('F'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('G'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('H'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('I'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('J'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('K'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('L'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('M'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('N'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('O'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('P'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('Q'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('R'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('S'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('T'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('U'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('V'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('W'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('X'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('Y'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('Z'), Integer.MAX_VALUE);

        System.out.println("Dijkstras starting from E");
        Map<Vertex<Character>, Integer> result = GraphAlgorithms.dijkstras(new Vertex<>('E'), largeGraph);
        assertEquals(expected, result); // Check amount & order of vertices returned
    }

    @Test(timeout = TIMEOUT)
    public void testDijkstrasLarge_F() {
        // Test starting at vertex F
        Map<Vertex<Character>, Integer> expected = new HashMap<>();
        expected.put(new Vertex<>('A'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('B'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('C'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('D'), 3);
        expected.put(new Vertex<>('E'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('F'), 0);
        expected.put(new Vertex<>('G'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('H'), 12);
        expected.put(new Vertex<>('I'), 11);
        expected.put(new Vertex<>('J'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('K'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('L'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('M'), 6);
        expected.put(new Vertex<>('N'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('O'), 1);
        expected.put(new Vertex<>('P'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('Q'), 5);
        expected.put(new Vertex<>('R'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('S'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('T'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('U'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('V'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('W'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('X'), 10);
        expected.put(new Vertex<>('Y'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('Z'), Integer.MAX_VALUE);

        System.out.println("Dijkstras starting from F");
        Map<Vertex<Character>, Integer> result = GraphAlgorithms.dijkstras(new Vertex<>('F'), largeGraph);
        assertEquals(expected, result); // Check amount & order of vertices returned
    }

    @Test(timeout = TIMEOUT)
    public void testDijkstrasLarge_G() {
        // Test starting at vertex G
        Map<Vertex<Character>, Integer> expected = new HashMap<>();
        expected.put(new Vertex<>('A'), 17);
        expected.put(new Vertex<>('B'), 17);
        expected.put(new Vertex<>('C'), 19);
        expected.put(new Vertex<>('D'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('E'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('F'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('G'), 0);
        expected.put(new Vertex<>('H'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('I'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('J'), 10);
        expected.put(new Vertex<>('K'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('L'), 19);
        expected.put(new Vertex<>('M'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('N'), 16);
        expected.put(new Vertex<>('O'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('P'), 5);
        expected.put(new Vertex<>('Q'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('R'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('S'), 23);
        expected.put(new Vertex<>('T'), 3);
        expected.put(new Vertex<>('U'), 9);
        expected.put(new Vertex<>('V'), 20);
        expected.put(new Vertex<>('W'), 3);
        expected.put(new Vertex<>('X'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('Y'), 18);
        expected.put(new Vertex<>('Z'), 21);

        System.out.println("Dijkstras starting from G");
        Map<Vertex<Character>, Integer> result = GraphAlgorithms.dijkstras(new Vertex<>('G'), largeGraph);
        assertEquals(expected, result); // Check amount & order of vertices returned
    }

    @Test(timeout = TIMEOUT)
    public void testDijkstrasLarge_H() {
        // Test starting at vertex H
        Map<Vertex<Character>, Integer> expected = new HashMap<>();
        expected.put(new Vertex<>('A'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('B'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('C'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('D'), 9);
        expected.put(new Vertex<>('E'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('F'), 12);
        expected.put(new Vertex<>('G'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('H'), 0);
        expected.put(new Vertex<>('I'), 21);
        expected.put(new Vertex<>('J'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('K'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('L'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('M'), 6);
        expected.put(new Vertex<>('N'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('O'), 11);
        expected.put(new Vertex<>('P'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('Q'), 15);
        expected.put(new Vertex<>('R'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('S'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('T'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('U'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('V'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('W'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('X'), 20);
        expected.put(new Vertex<>('Y'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('Z'), Integer.MAX_VALUE);

        System.out.println("Dijkstras starting from H");
        Map<Vertex<Character>, Integer> result = GraphAlgorithms.dijkstras(new Vertex<>('H'), largeGraph);
        assertEquals(expected, result); // Check amount & order of vertices returned
    }

    @Test(timeout = TIMEOUT)
    public void testDijkstrasLarge_I() {
        // Test starting at vertex I
        Map<Vertex<Character>, Integer> expected = new HashMap<>();
        expected.put(new Vertex<>('A'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('B'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('C'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('D'), 12);
        expected.put(new Vertex<>('E'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('F'), 11);
        expected.put(new Vertex<>('G'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('H'), 21);
        expected.put(new Vertex<>('I'), 0);
        expected.put(new Vertex<>('J'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('K'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('L'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('M'), 15);
        expected.put(new Vertex<>('N'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('O'), 10);
        expected.put(new Vertex<>('P'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('Q'), 6);
        expected.put(new Vertex<>('R'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('S'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('T'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('U'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('V'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('W'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('X'), 7);
        expected.put(new Vertex<>('Y'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('Z'), Integer.MAX_VALUE);

        System.out.println("Dijkstras starting from I");
        Map<Vertex<Character>, Integer> result = GraphAlgorithms.dijkstras(new Vertex<>('I'), largeGraph);
        assertEquals(expected, result); // Check amount & order of vertices returned
    }

    @Test(timeout = TIMEOUT)
    public void testDijkstrasLarge_J() {
        // Test starting at vertex J
        Map<Vertex<Character>, Integer> expected = new HashMap<>();
        expected.put(new Vertex<>('A'), 7);
        expected.put(new Vertex<>('B'), 7);
        expected.put(new Vertex<>('C'), 9);
        expected.put(new Vertex<>('D'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('E'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('F'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('G'), 10);
        expected.put(new Vertex<>('H'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('I'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('J'), 0);
        expected.put(new Vertex<>('K'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('L'), 9);
        expected.put(new Vertex<>('M'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('N'), 6);
        expected.put(new Vertex<>('O'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('P'), 5);
        expected.put(new Vertex<>('Q'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('R'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('S'), 13);
        expected.put(new Vertex<>('T'), 7);
        expected.put(new Vertex<>('U'), 1);
        expected.put(new Vertex<>('V'), 10);
        expected.put(new Vertex<>('W'), 11);
        expected.put(new Vertex<>('X'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('Y'), 8);
        expected.put(new Vertex<>('Z'), 11);

        System.out.println("Dijkstras starting from J");
        Map<Vertex<Character>, Integer> result = GraphAlgorithms.dijkstras(new Vertex<>('J'), largeGraph);
        assertEquals(expected, result); // Check amount & order of vertices returned
    }

    @Test(timeout = TIMEOUT)
    public void testDijkstrasLarge_K() {
        // Test starting at vertex K
        Map<Vertex<Character>, Integer> expected = new HashMap<>();
        expected.put(new Vertex<>('A'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('B'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('C'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('D'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('E'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('F'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('G'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('H'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('I'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('J'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('K'), 0);
        expected.put(new Vertex<>('L'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('M'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('N'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('O'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('P'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('Q'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('R'), 9);
        expected.put(new Vertex<>('S'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('T'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('U'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('V'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('W'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('X'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('Y'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('Z'), Integer.MAX_VALUE);

        System.out.println("Dijkstras starting from K");
        Map<Vertex<Character>, Integer> result = GraphAlgorithms.dijkstras(new Vertex<>('K'), largeGraph);
        assertEquals(expected, result); // Check amount & order of vertices returned
    }

    @Test(timeout = TIMEOUT)
    public void testDijkstrasLarge_L() {
        // Test starting at vertex L
        Map<Vertex<Character>, Integer> expected = new HashMap<>();
        expected.put(new Vertex<>('A'), 2);
        expected.put(new Vertex<>('B'), 2);
        expected.put(new Vertex<>('C'), 0);
        expected.put(new Vertex<>('D'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('E'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('F'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('G'), 19);
        expected.put(new Vertex<>('H'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('I'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('J'), 9);
        expected.put(new Vertex<>('K'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('L'), 0);
        expected.put(new Vertex<>('M'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('N'), 12);
        expected.put(new Vertex<>('O'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('P'), 14);
        expected.put(new Vertex<>('Q'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('R'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('S'), 19);
        expected.put(new Vertex<>('T'), 16);
        expected.put(new Vertex<>('U'), 10);
        expected.put(new Vertex<>('V'), 5);
        expected.put(new Vertex<>('W'), 20);
        expected.put(new Vertex<>('X'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('Y'), 3);
        expected.put(new Vertex<>('Z'), 3);

        System.out.println("Dijkstras starting from L");
        Map<Vertex<Character>, Integer> result = GraphAlgorithms.dijkstras(new Vertex<>('L'), largeGraph);
        assertEquals(expected, result); // Check amount & order of vertices returned
    }

    @Test(timeout = TIMEOUT)
    public void testDijkstrasLarge_M() {
        // Test starting at vertex M
        Map<Vertex<Character>, Integer> expected = new HashMap<>();
        expected.put(new Vertex<>('A'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('B'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('C'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('D'), 3);
        expected.put(new Vertex<>('E'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('F'), 6);
        expected.put(new Vertex<>('G'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('H'), 6);
        expected.put(new Vertex<>('I'), 15);
        expected.put(new Vertex<>('J'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('K'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('L'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('M'), 0);
        expected.put(new Vertex<>('N'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('O'), 5);
        expected.put(new Vertex<>('P'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('Q'), 9);
        expected.put(new Vertex<>('R'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('S'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('T'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('U'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('V'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('W'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('X'), 14);
        expected.put(new Vertex<>('Y'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('Z'), Integer.MAX_VALUE);

        System.out.println("Dijkstras starting from M");
        Map<Vertex<Character>, Integer> result = GraphAlgorithms.dijkstras(new Vertex<>('M'), largeGraph);
        assertEquals(expected, result); // Check amount & order of vertices returned
    }

    @Test(timeout = TIMEOUT)
    public void testDijkstrasLarge_N() {
        // Test starting at vertex N
        Map<Vertex<Character>, Integer> expected = new HashMap<>();
        expected.put(new Vertex<>('A'), 10);
        expected.put(new Vertex<>('B'), 10);
        expected.put(new Vertex<>('C'), 12);
        expected.put(new Vertex<>('D'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('E'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('F'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('G'), 16);
        expected.put(new Vertex<>('H'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('I'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('J'), 6);
        expected.put(new Vertex<>('K'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('L'), 12);
        expected.put(new Vertex<>('M'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('N'), 0);
        expected.put(new Vertex<>('O'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('P'), 11);
        expected.put(new Vertex<>('Q'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('R'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('S'), 7);
        expected.put(new Vertex<>('T'), 13);
        expected.put(new Vertex<>('U'), 7);
        expected.put(new Vertex<>('V'), 13);
        expected.put(new Vertex<>('W'), 17);
        expected.put(new Vertex<>('X'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('Y'), 9);
        expected.put(new Vertex<>('Z'), 14);

        System.out.println("Dijkstras starting from N");
        Map<Vertex<Character>, Integer> result = GraphAlgorithms.dijkstras(new Vertex<>('N'), largeGraph);
        assertEquals(expected, result); // Check amount & order of vertices returned
    }

    @Test(timeout = TIMEOUT)
    public void testDijkstrasLarge_O() {
        // Test starting at vertex O
        Map<Vertex<Character>, Integer> expected = new HashMap<>();
        expected.put(new Vertex<>('A'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('B'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('C'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('D'), 2);
        expected.put(new Vertex<>('E'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('F'), 1);
        expected.put(new Vertex<>('G'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('H'), 11);
        expected.put(new Vertex<>('I'), 10);
        expected.put(new Vertex<>('J'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('K'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('L'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('M'), 5);
        expected.put(new Vertex<>('N'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('O'), 0);
        expected.put(new Vertex<>('P'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('Q'), 4);
        expected.put(new Vertex<>('R'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('S'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('T'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('U'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('V'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('W'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('X'), 9);
        expected.put(new Vertex<>('Y'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('Z'), Integer.MAX_VALUE);

        System.out.println("Dijkstras starting from O");
        Map<Vertex<Character>, Integer> result = GraphAlgorithms.dijkstras(new Vertex<>('O'), largeGraph);
        assertEquals(expected, result); // Check amount & order of vertices returned
    }

    @Test(timeout = TIMEOUT)
    public void testDijkstrasLarge_P() {
        // Test starting at vertex P
        Map<Vertex<Character>, Integer> expected = new HashMap<>();
        expected.put(new Vertex<>('A'), 12);
        expected.put(new Vertex<>('B'), 12);
        expected.put(new Vertex<>('C'), 14);
        expected.put(new Vertex<>('D'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('E'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('F'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('G'), 5);
        expected.put(new Vertex<>('H'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('I'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('J'), 5);
        expected.put(new Vertex<>('K'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('L'), 14);
        expected.put(new Vertex<>('M'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('N'), 11);
        expected.put(new Vertex<>('O'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('P'), 0);
        expected.put(new Vertex<>('Q'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('R'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('S'), 18);
        expected.put(new Vertex<>('T'), 2);
        expected.put(new Vertex<>('U'), 4);
        expected.put(new Vertex<>('V'), 15);
        expected.put(new Vertex<>('W'), 6);
        expected.put(new Vertex<>('X'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('Y'), 13);
        expected.put(new Vertex<>('Z'), 16);

        System.out.println("Dijkstras starting from P");
        Map<Vertex<Character>, Integer> result = GraphAlgorithms.dijkstras(new Vertex<>('P'), largeGraph);
        assertEquals(expected, result); // Check amount & order of vertices returned
    }

    @Test(timeout = TIMEOUT)
    public void testDijkstrasLarge_Q() {
        // Test starting at vertex Q
        Map<Vertex<Character>, Integer> expected = new HashMap<>();
        expected.put(new Vertex<>('A'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('B'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('C'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('D'), 6);
        expected.put(new Vertex<>('E'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('F'), 5);
        expected.put(new Vertex<>('G'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('H'), 15);
        expected.put(new Vertex<>('I'), 6);
        expected.put(new Vertex<>('J'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('K'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('L'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('M'), 9);
        expected.put(new Vertex<>('N'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('O'), 4);
        expected.put(new Vertex<>('P'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('Q'), 0);
        expected.put(new Vertex<>('R'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('S'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('T'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('U'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('V'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('W'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('X'), 5);
        expected.put(new Vertex<>('Y'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('Z'), Integer.MAX_VALUE);

        System.out.println("Dijkstras starting from Q");
        Map<Vertex<Character>, Integer> result = GraphAlgorithms.dijkstras(new Vertex<>('Q'), largeGraph);
        assertEquals(expected, result); // Check amount & order of vertices returned
    }

    @Test(timeout = TIMEOUT)
    public void testDijkstrasLarge_R() {
        // Test starting at vertex R
        Map<Vertex<Character>, Integer> expected = new HashMap<>();
        expected.put(new Vertex<>('A'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('B'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('C'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('D'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('E'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('F'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('G'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('H'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('I'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('J'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('K'), 9);
        expected.put(new Vertex<>('L'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('M'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('N'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('O'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('P'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('Q'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('R'), 0);
        expected.put(new Vertex<>('S'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('T'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('U'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('V'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('W'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('X'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('Y'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('Z'), Integer.MAX_VALUE);

        System.out.println("Dijkstras starting from R");
        Map<Vertex<Character>, Integer> result = GraphAlgorithms.dijkstras(new Vertex<>('R'), largeGraph);
        assertEquals(expected, result); // Check amount & order of vertices returned
    }

    @Test(timeout = TIMEOUT)
    public void testDijkstrasLarge_S() {
        // Test starting at vertex S
        Map<Vertex<Character>, Integer> expected = new HashMap<>();
        expected.put(new Vertex<>('A'), 17);
        expected.put(new Vertex<>('B'), 17);
        expected.put(new Vertex<>('C'), 19);
        expected.put(new Vertex<>('D'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('E'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('F'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('G'), 23);
        expected.put(new Vertex<>('H'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('I'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('J'), 13);
        expected.put(new Vertex<>('K'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('L'), 19);
        expected.put(new Vertex<>('M'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('N'), 7);
        expected.put(new Vertex<>('O'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('P'), 18);
        expected.put(new Vertex<>('Q'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('R'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('S'), 0);
        expected.put(new Vertex<>('T'), 20);
        expected.put(new Vertex<>('U'), 14);
        expected.put(new Vertex<>('V'), 20);
        expected.put(new Vertex<>('W'), 24);
        expected.put(new Vertex<>('X'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('Y'), 16);
        expected.put(new Vertex<>('Z'), 21);

        System.out.println("Dijkstras starting from S");
        Map<Vertex<Character>, Integer> result = GraphAlgorithms.dijkstras(new Vertex<>('S'), largeGraph);
        assertEquals(expected, result); // Check amount & order of vertices returned
    }

    @Test(timeout = TIMEOUT)
    public void testDijkstrasLarge_T() {
        // Test starting at vertex T
        Map<Vertex<Character>, Integer> expected = new HashMap<>();
        expected.put(new Vertex<>('A'), 14);
        expected.put(new Vertex<>('B'), 14);
        expected.put(new Vertex<>('C'), 16);
        expected.put(new Vertex<>('D'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('E'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('F'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('G'), 3);
        expected.put(new Vertex<>('H'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('I'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('J'), 7);
        expected.put(new Vertex<>('K'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('L'), 16);
        expected.put(new Vertex<>('M'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('N'), 13);
        expected.put(new Vertex<>('O'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('P'), 2);
        expected.put(new Vertex<>('Q'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('R'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('S'), 20);
        expected.put(new Vertex<>('T'), 0);
        expected.put(new Vertex<>('U'), 6);
        expected.put(new Vertex<>('V'), 17);
        expected.put(new Vertex<>('W'), 6);
        expected.put(new Vertex<>('X'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('Y'), 15);
        expected.put(new Vertex<>('Z'), 18);

        System.out.println("Dijkstras starting from T");
        Map<Vertex<Character>, Integer> result = GraphAlgorithms.dijkstras(new Vertex<>('T'), largeGraph);
        assertEquals(expected, result); // Check amount & order of vertices returned
    }

    @Test(timeout = TIMEOUT)
    public void testDijkstrasLarge_U() {
        // Test starting at vertex U
        Map<Vertex<Character>, Integer> expected = new HashMap<>();
        expected.put(new Vertex<>('A'), 8);
        expected.put(new Vertex<>('B'), 8);
        expected.put(new Vertex<>('C'), 10);
        expected.put(new Vertex<>('D'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('E'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('F'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('G'), 9);
        expected.put(new Vertex<>('H'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('I'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('J'), 1);
        expected.put(new Vertex<>('K'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('L'), 10);
        expected.put(new Vertex<>('M'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('N'), 7);
        expected.put(new Vertex<>('O'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('P'), 4);
        expected.put(new Vertex<>('Q'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('R'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('S'), 14);
        expected.put(new Vertex<>('T'), 6);
        expected.put(new Vertex<>('U'), 0);
        expected.put(new Vertex<>('V'), 11);
        expected.put(new Vertex<>('W'), 10);
        expected.put(new Vertex<>('X'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('Y'), 9);
        expected.put(new Vertex<>('Z'), 12);

        System.out.println("Dijkstras starting from U");
        Map<Vertex<Character>, Integer> result = GraphAlgorithms.dijkstras(new Vertex<>('U'), largeGraph);
        assertEquals(expected, result); // Check amount & order of vertices returned
    }

    @Test(timeout = TIMEOUT)
    public void testDijkstrasLarge_V() {
        // Test starting at vertex V
        Map<Vertex<Character>, Integer> expected = new HashMap<>();
        expected.put(new Vertex<>('A'), 3);
        expected.put(new Vertex<>('B'), 3);
        expected.put(new Vertex<>('C'), 5);
        expected.put(new Vertex<>('D'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('E'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('F'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('G'), 20);
        expected.put(new Vertex<>('H'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('I'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('J'), 10);
        expected.put(new Vertex<>('K'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('L'), 5);
        expected.put(new Vertex<>('M'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('N'), 13);
        expected.put(new Vertex<>('O'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('P'), 15);
        expected.put(new Vertex<>('Q'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('R'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('S'), 20);
        expected.put(new Vertex<>('T'), 17);
        expected.put(new Vertex<>('U'), 11);
        expected.put(new Vertex<>('V'), 0);
        expected.put(new Vertex<>('W'), 21);
        expected.put(new Vertex<>('X'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('Y'), 4);
        expected.put(new Vertex<>('Z'), 7);

        System.out.println("Dijkstras starting from V");
        Map<Vertex<Character>, Integer> result = GraphAlgorithms.dijkstras(new Vertex<>('V'), largeGraph);
        assertEquals(expected, result); // Check amount & order of vertices returned
    }

    @Test(timeout = TIMEOUT)
    public void testDijkstrasLarge_W() {
        // Test starting at vertex W
        Map<Vertex<Character>, Integer> expected = new HashMap<>();
        expected.put(new Vertex<>('A'), 18);
        expected.put(new Vertex<>('B'), 18);
        expected.put(new Vertex<>('C'), 20);
        expected.put(new Vertex<>('D'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('E'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('F'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('G'), 3);
        expected.put(new Vertex<>('H'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('I'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('J'), 11);
        expected.put(new Vertex<>('K'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('L'), 20);
        expected.put(new Vertex<>('M'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('N'), 17);
        expected.put(new Vertex<>('O'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('P'), 6);
        expected.put(new Vertex<>('Q'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('R'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('S'), 24);
        expected.put(new Vertex<>('T'), 6);
        expected.put(new Vertex<>('U'), 10);
        expected.put(new Vertex<>('V'), 21);
        expected.put(new Vertex<>('W'), 0);
        expected.put(new Vertex<>('X'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('Y'), 19);
        expected.put(new Vertex<>('Z'), 22);

        System.out.println("Dijkstras starting from W");
        Map<Vertex<Character>, Integer> result = GraphAlgorithms.dijkstras(new Vertex<>('W'), largeGraph);
        assertEquals(expected, result); // Check amount & order of vertices returned
    }

    @Test(timeout = TIMEOUT)
    public void testDijkstrasLarge_X() {
        // Test starting at vertex X
        Map<Vertex<Character>, Integer> expected = new HashMap<>();
        expected.put(new Vertex<>('A'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('B'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('C'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('D'), 11);
        expected.put(new Vertex<>('E'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('F'), 10);
        expected.put(new Vertex<>('G'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('H'), 20);
        expected.put(new Vertex<>('I'), 7);
        expected.put(new Vertex<>('J'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('K'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('L'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('M'), 14);
        expected.put(new Vertex<>('N'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('O'), 9);
        expected.put(new Vertex<>('P'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('Q'), 5);
        expected.put(new Vertex<>('R'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('S'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('T'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('U'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('V'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('W'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('X'), 0);
        expected.put(new Vertex<>('Y'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('Z'), Integer.MAX_VALUE);

        System.out.println("Dijkstras starting from X");
        Map<Vertex<Character>, Integer> result = GraphAlgorithms.dijkstras(new Vertex<>('X'), largeGraph);
        assertEquals(expected, result); // Check amount & order of vertices returned
    }

    @Test(timeout = TIMEOUT)
    public void testDijkstrasLarge_Y() {
        // Test starting at vertex Y
        Map<Vertex<Character>, Integer> expected = new HashMap<>();
        expected.put(new Vertex<>('A'), 1);
        expected.put(new Vertex<>('B'), 1);
        expected.put(new Vertex<>('C'), 3);
        expected.put(new Vertex<>('D'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('E'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('F'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('G'), 18);
        expected.put(new Vertex<>('H'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('I'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('J'), 8);
        expected.put(new Vertex<>('K'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('L'), 3);
        expected.put(new Vertex<>('M'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('N'), 9);
        expected.put(new Vertex<>('O'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('P'), 13);
        expected.put(new Vertex<>('Q'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('R'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('S'), 16);
        expected.put(new Vertex<>('T'), 15);
        expected.put(new Vertex<>('U'), 9);
        expected.put(new Vertex<>('V'), 4);
        expected.put(new Vertex<>('W'), 19);
        expected.put(new Vertex<>('X'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('Y'), 0);
        expected.put(new Vertex<>('Z'), 5);

        System.out.println("Dijkstras starting from Y");
        Map<Vertex<Character>, Integer> result = GraphAlgorithms.dijkstras(new Vertex<>('Y'), largeGraph);
        assertEquals(expected, result); // Check amount & order of vertices returned
    }

    @Test(timeout = TIMEOUT)
    public void testDijkstrasLarge_Z() {
        // Test starting at vertex Z
        Map<Vertex<Character>, Integer> expected = new HashMap<>();
        expected.put(new Vertex<>('A'), 4);
        expected.put(new Vertex<>('B'), 4);
        expected.put(new Vertex<>('C'), 3);
        expected.put(new Vertex<>('D'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('E'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('F'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('G'), 21);
        expected.put(new Vertex<>('H'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('I'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('J'), 11);
        expected.put(new Vertex<>('K'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('L'), 3);
        expected.put(new Vertex<>('M'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('N'), 14);
        expected.put(new Vertex<>('O'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('P'), 16);
        expected.put(new Vertex<>('Q'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('R'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('S'), 21);
        expected.put(new Vertex<>('T'), 18);
        expected.put(new Vertex<>('U'), 12);
        expected.put(new Vertex<>('V'), 7);
        expected.put(new Vertex<>('W'), 22);
        expected.put(new Vertex<>('X'), Integer.MAX_VALUE);
        expected.put(new Vertex<>('Y'), 5);
        expected.put(new Vertex<>('Z'), 0);

        System.out.println("Dijkstras starting from Z");
        Map<Vertex<Character>, Integer> result = GraphAlgorithms.dijkstras(new Vertex<>('Z'), largeGraph);
        assertEquals(expected, result); // Check amount & order of vertices returned
    }

    @Test(timeout = TIMEOUT)
    public void testBFSLarge_A() {
        // Test starting at vertex A
        List<Vertex<Character>> expected = new LinkedList<>();
        expected.add(new Vertex<>('A'));
        expected.add(new Vertex<>('B'));
        expected.add(new Vertex<>('C'));
        expected.add(new Vertex<>('J'));
        expected.add(new Vertex<>('V'));
        expected.add(new Vertex<>('Y'));
        expected.add(new Vertex<>('Z'));
        expected.add(new Vertex<>('L'));
        expected.add(new Vertex<>('N'));
        expected.add(new Vertex<>('U'));
        expected.add(new Vertex<>('S'));
        expected.add(new Vertex<>('P'));
        expected.add(new Vertex<>('T'));
        expected.add(new Vertex<>('W'));
        expected.add(new Vertex<>('G'));

        System.out.println("BFS starting from A");
        List<Vertex<Character>> result = GraphAlgorithms.bfs(new Vertex<>('A'), largeGraph);
        assertEquals(expected, result); // Check amount & order of vertices returned
    }

    @Test(timeout = TIMEOUT)
    public void testBFSLarge_B() {
        // Test starting at vertex B
        List<Vertex<Character>> expected = new LinkedList<>();
        expected.add(new Vertex<>('B'));
        expected.add(new Vertex<>('A'));
        expected.add(new Vertex<>('Y'));
        expected.add(new Vertex<>('Z'));
        expected.add(new Vertex<>('C'));
        expected.add(new Vertex<>('J'));
        expected.add(new Vertex<>('V'));
        expected.add(new Vertex<>('N'));
        expected.add(new Vertex<>('L'));
        expected.add(new Vertex<>('U'));
        expected.add(new Vertex<>('S'));
        expected.add(new Vertex<>('P'));
        expected.add(new Vertex<>('T'));
        expected.add(new Vertex<>('W'));
        expected.add(new Vertex<>('G'));

        System.out.println("BFS starting from B");
        List<Vertex<Character>> result = GraphAlgorithms.bfs(new Vertex<>('B'), largeGraph);
        assertEquals(expected, result); // Check amount & order of vertices returned
    }

    @Test(timeout = TIMEOUT)
    public void testBFSLarge_C() {
        // Test starting at vertex C
        List<Vertex<Character>> expected = new LinkedList<>();
        expected.add(new Vertex<>('C'));
        expected.add(new Vertex<>('A'));
        expected.add(new Vertex<>('L'));
        expected.add(new Vertex<>('Z'));
        expected.add(new Vertex<>('B'));
        expected.add(new Vertex<>('J'));
        expected.add(new Vertex<>('V'));
        expected.add(new Vertex<>('Y'));
        expected.add(new Vertex<>('N'));
        expected.add(new Vertex<>('U'));
        expected.add(new Vertex<>('S'));
        expected.add(new Vertex<>('P'));
        expected.add(new Vertex<>('T'));
        expected.add(new Vertex<>('W'));
        expected.add(new Vertex<>('G'));

        System.out.println("BFS starting from C");
        List<Vertex<Character>> result = GraphAlgorithms.bfs(new Vertex<>('C'), largeGraph);
        assertEquals(expected, result); // Check amount & order of vertices returned
    }

    @Test(timeout = TIMEOUT)
    public void testBFSLarge_D() {
        // Test starting at vertex D
        List<Vertex<Character>> expected = new LinkedList<>();
        expected.add(new Vertex<>('D'));
        expected.add(new Vertex<>('M'));
        expected.add(new Vertex<>('O'));
        expected.add(new Vertex<>('F'));
        expected.add(new Vertex<>('H'));
        expected.add(new Vertex<>('Q'));
        expected.add(new Vertex<>('I'));
        expected.add(new Vertex<>('X'));

        System.out.println("BFS starting from D");
        List<Vertex<Character>> result = GraphAlgorithms.bfs(new Vertex<>('D'), largeGraph);
        assertEquals(expected, result); // Check amount & order of vertices returned
    }

    @Test(timeout = TIMEOUT)
    public void testBFSLarge_E() {
        // Test starting at vertex E
        List<Vertex<Character>> expected = new LinkedList<>();
        expected.add(new Vertex<>('E'));

        System.out.println("BFS starting from E");
        List<Vertex<Character>> result = GraphAlgorithms.bfs(new Vertex<>('E'), largeGraph);
        assertEquals(expected, result); // Check amount & order of vertices returned
    }

    @Test(timeout = TIMEOUT)
    public void testBFSLarge_F() {
        // Test starting at vertex F
        List<Vertex<Character>> expected = new LinkedList<>();
        expected.add(new Vertex<>('F'));
        expected.add(new Vertex<>('M'));
        expected.add(new Vertex<>('O'));
        expected.add(new Vertex<>('D'));
        expected.add(new Vertex<>('H'));
        expected.add(new Vertex<>('Q'));
        expected.add(new Vertex<>('I'));
        expected.add(new Vertex<>('X'));

        System.out.println("BFS starting from F");
        List<Vertex<Character>> result = GraphAlgorithms.bfs(new Vertex<>('F'), largeGraph);
        assertEquals(expected, result); // Check amount & order of vertices returned
    }

    @Test(timeout = TIMEOUT)
    public void testBFSLarge_G() {
        // Test starting at vertex G
        List<Vertex<Character>> expected = new LinkedList<>();
        expected.add(new Vertex<>('G'));
        expected.add(new Vertex<>('T'));
        expected.add(new Vertex<>('W'));
        expected.add(new Vertex<>('P'));
        expected.add(new Vertex<>('U'));
        expected.add(new Vertex<>('J'));
        expected.add(new Vertex<>('A'));
        expected.add(new Vertex<>('N'));
        expected.add(new Vertex<>('B'));
        expected.add(new Vertex<>('C'));
        expected.add(new Vertex<>('V'));
        expected.add(new Vertex<>('S'));
        expected.add(new Vertex<>('Y'));
        expected.add(new Vertex<>('Z'));
        expected.add(new Vertex<>('L'));

        System.out.println("BFS starting from G");
        List<Vertex<Character>> result = GraphAlgorithms.bfs(new Vertex<>('G'), largeGraph);
        assertEquals(expected, result); // Check amount & order of vertices returned
    }

    @Test(timeout = TIMEOUT)
    public void testBFSLarge_H() {
        // Test starting at vertex H
        List<Vertex<Character>> expected = new LinkedList<>();
        expected.add(new Vertex<>('H'));
        expected.add(new Vertex<>('M'));
        expected.add(new Vertex<>('D'));
        expected.add(new Vertex<>('F'));
        expected.add(new Vertex<>('O'));
        expected.add(new Vertex<>('Q'));
        expected.add(new Vertex<>('I'));
        expected.add(new Vertex<>('X'));

        System.out.println("BFS starting from H");
        List<Vertex<Character>> result = GraphAlgorithms.bfs(new Vertex<>('H'), largeGraph);
        assertEquals(expected, result); // Check amount & order of vertices returned
    }

    @Test(timeout = TIMEOUT)
    public void testBFSLarge_I() {
        // Test starting at vertex I
        List<Vertex<Character>> expected = new LinkedList<>();
        expected.add(new Vertex<>('I'));
        expected.add(new Vertex<>('Q'));
        expected.add(new Vertex<>('X'));
        expected.add(new Vertex<>('O'));
        expected.add(new Vertex<>('D'));
        expected.add(new Vertex<>('F'));
        expected.add(new Vertex<>('M'));
        expected.add(new Vertex<>('H'));

        System.out.println("BFS starting from I");
        List<Vertex<Character>> result = GraphAlgorithms.bfs(new Vertex<>('I'), largeGraph);
        assertEquals(expected, result); // Check amount & order of vertices returned
    }

    @Test(timeout = TIMEOUT)
    public void testBFSLarge_J() {
        // Test starting at vertex J
        List<Vertex<Character>> expected = new LinkedList<>();
        expected.add(new Vertex<>('J'));
        expected.add(new Vertex<>('A'));
        expected.add(new Vertex<>('N'));
        expected.add(new Vertex<>('U'));
        expected.add(new Vertex<>('B'));
        expected.add(new Vertex<>('C'));
        expected.add(new Vertex<>('V'));
        expected.add(new Vertex<>('S'));
        expected.add(new Vertex<>('Y'));
        expected.add(new Vertex<>('P'));
        expected.add(new Vertex<>('Z'));
        expected.add(new Vertex<>('L'));
        expected.add(new Vertex<>('T'));
        expected.add(new Vertex<>('W'));
        expected.add(new Vertex<>('G'));

        System.out.println("BFS starting from J");
        List<Vertex<Character>> result = GraphAlgorithms.bfs(new Vertex<>('J'), largeGraph);
        assertEquals(expected, result); // Check amount & order of vertices returned
    }

    @Test(timeout = TIMEOUT)
    public void testBFSLarge_K() {
        // Test starting at vertex K
        List<Vertex<Character>> expected = new LinkedList<>();
        expected.add(new Vertex<>('K'));
        expected.add(new Vertex<>('R'));

        System.out.println("BFS starting from K");
        List<Vertex<Character>> result = GraphAlgorithms.bfs(new Vertex<>('K'), largeGraph);
        assertEquals(expected, result); // Check amount & order of vertices returned
    }

    @Test(timeout = TIMEOUT)
    public void testBFSLarge_L() {
        // Test starting at vertex L
        List<Vertex<Character>> expected = new LinkedList<>();
        expected.add(new Vertex<>('L'));
        expected.add(new Vertex<>('C'));
        expected.add(new Vertex<>('A'));
        expected.add(new Vertex<>('Z'));
        expected.add(new Vertex<>('B'));
        expected.add(new Vertex<>('J'));
        expected.add(new Vertex<>('V'));
        expected.add(new Vertex<>('Y'));
        expected.add(new Vertex<>('N'));
        expected.add(new Vertex<>('U'));
        expected.add(new Vertex<>('S'));
        expected.add(new Vertex<>('P'));
        expected.add(new Vertex<>('T'));
        expected.add(new Vertex<>('W'));
        expected.add(new Vertex<>('G'));

        System.out.println("BFS starting from L");
        List<Vertex<Character>> result = GraphAlgorithms.bfs(new Vertex<>('L'), largeGraph);
        assertEquals(expected, result); // Check amount & order of vertices returned
    }

    @Test(timeout = TIMEOUT)
    public void testBFSLarge_M() {
        // Test starting at vertex M
        List<Vertex<Character>> expected = new LinkedList<>();
        expected.add(new Vertex<>('M'));
        expected.add(new Vertex<>('D'));
        expected.add(new Vertex<>('F'));
        expected.add(new Vertex<>('H'));
        expected.add(new Vertex<>('O'));
        expected.add(new Vertex<>('Q'));
        expected.add(new Vertex<>('I'));
        expected.add(new Vertex<>('X'));

        System.out.println("BFS starting from M");
        List<Vertex<Character>> result = GraphAlgorithms.bfs(new Vertex<>('M'), largeGraph);
        assertEquals(expected, result); // Check amount & order of vertices returned
    }

    @Test(timeout = TIMEOUT)
    public void testBFSLarge_N() {
        // Test starting at vertex N
        List<Vertex<Character>> expected = new LinkedList<>();
        expected.add(new Vertex<>('N'));
        expected.add(new Vertex<>('J'));
        expected.add(new Vertex<>('S'));
        expected.add(new Vertex<>('Y'));
        expected.add(new Vertex<>('A'));
        expected.add(new Vertex<>('U'));
        expected.add(new Vertex<>('B'));
        expected.add(new Vertex<>('Z'));
        expected.add(new Vertex<>('C'));
        expected.add(new Vertex<>('V'));
        expected.add(new Vertex<>('P'));
        expected.add(new Vertex<>('L'));
        expected.add(new Vertex<>('T'));
        expected.add(new Vertex<>('W'));
        expected.add(new Vertex<>('G'));

        System.out.println("BFS starting from N");
        List<Vertex<Character>> result = GraphAlgorithms.bfs(new Vertex<>('N'), largeGraph);
        assertEquals(expected, result); // Check amount & order of vertices returned
    }

    @Test(timeout = TIMEOUT)
    public void testBFSLarge_O() {
        // Test starting at vertex O
        List<Vertex<Character>> expected = new LinkedList<>();
        expected.add(new Vertex<>('O'));
        expected.add(new Vertex<>('D'));
        expected.add(new Vertex<>('F'));
        expected.add(new Vertex<>('M'));
        expected.add(new Vertex<>('Q'));
        expected.add(new Vertex<>('H'));
        expected.add(new Vertex<>('I'));
        expected.add(new Vertex<>('X'));

        System.out.println("BFS starting from O");
        List<Vertex<Character>> result = GraphAlgorithms.bfs(new Vertex<>('O'), largeGraph);
        assertEquals(expected, result); // Check amount & order of vertices returned
    }

    @Test(timeout = TIMEOUT)
    public void testBFSLarge_P() {
        // Test starting at vertex P
        List<Vertex<Character>> expected = new LinkedList<>();
        expected.add(new Vertex<>('P'));
        expected.add(new Vertex<>('T'));
        expected.add(new Vertex<>('U'));
        expected.add(new Vertex<>('W'));
        expected.add(new Vertex<>('G'));
        expected.add(new Vertex<>('J'));
        expected.add(new Vertex<>('A'));
        expected.add(new Vertex<>('N'));
        expected.add(new Vertex<>('B'));
        expected.add(new Vertex<>('C'));
        expected.add(new Vertex<>('V'));
        expected.add(new Vertex<>('S'));
        expected.add(new Vertex<>('Y'));
        expected.add(new Vertex<>('Z'));
        expected.add(new Vertex<>('L'));

        System.out.println("BFS starting from P");
        List<Vertex<Character>> result = GraphAlgorithms.bfs(new Vertex<>('P'), largeGraph);
        assertEquals(expected, result); // Check amount & order of vertices returned
    }

    @Test(timeout = TIMEOUT)
    public void testBFSLarge_Q() {
        // Test starting at vertex Q
        List<Vertex<Character>> expected = new LinkedList<>();
        expected.add(new Vertex<>('Q'));
        expected.add(new Vertex<>('I'));
        expected.add(new Vertex<>('O'));
        expected.add(new Vertex<>('X'));
        expected.add(new Vertex<>('D'));
        expected.add(new Vertex<>('F'));
        expected.add(new Vertex<>('M'));
        expected.add(new Vertex<>('H'));

        System.out.println("BFS starting from Q");
        List<Vertex<Character>> result = GraphAlgorithms.bfs(new Vertex<>('Q'), largeGraph);
        assertEquals(expected, result); // Check amount & order of vertices returned
    }

    @Test(timeout = TIMEOUT)
    public void testBFSLarge_R() {
        // Test starting at vertex R
        List<Vertex<Character>> expected = new LinkedList<>();
        expected.add(new Vertex<>('R'));
        expected.add(new Vertex<>('K'));

        System.out.println("BFS starting from R");
        List<Vertex<Character>> result = GraphAlgorithms.bfs(new Vertex<>('R'), largeGraph);
        assertEquals(expected, result); // Check amount & order of vertices returned
    }

    @Test(timeout = TIMEOUT)
    public void testBFSLarge_S() {
        // Test starting at vertex S
        List<Vertex<Character>> expected = new LinkedList<>();
        expected.add(new Vertex<>('S'));
        expected.add(new Vertex<>('N'));
        expected.add(new Vertex<>('J'));
        expected.add(new Vertex<>('Y'));
        expected.add(new Vertex<>('A'));
        expected.add(new Vertex<>('U'));
        expected.add(new Vertex<>('B'));
        expected.add(new Vertex<>('Z'));
        expected.add(new Vertex<>('C'));
        expected.add(new Vertex<>('V'));
        expected.add(new Vertex<>('P'));
        expected.add(new Vertex<>('L'));
        expected.add(new Vertex<>('T'));
        expected.add(new Vertex<>('W'));
        expected.add(new Vertex<>('G'));

        System.out.println("BFS starting from S");
        List<Vertex<Character>> result = GraphAlgorithms.bfs(new Vertex<>('S'), largeGraph);
        assertEquals(expected, result); // Check amount & order of vertices returned
    }

    @Test(timeout = TIMEOUT)
    public void testBFSLarge_T() {
        // Test starting at vertex T
        List<Vertex<Character>> expected = new LinkedList<>();
        expected.add(new Vertex<>('T'));
        expected.add(new Vertex<>('G'));
        expected.add(new Vertex<>('P'));
        expected.add(new Vertex<>('W'));
        expected.add(new Vertex<>('U'));
        expected.add(new Vertex<>('J'));
        expected.add(new Vertex<>('A'));
        expected.add(new Vertex<>('N'));
        expected.add(new Vertex<>('B'));
        expected.add(new Vertex<>('C'));
        expected.add(new Vertex<>('V'));
        expected.add(new Vertex<>('S'));
        expected.add(new Vertex<>('Y'));
        expected.add(new Vertex<>('Z'));
        expected.add(new Vertex<>('L'));

        System.out.println("BFS starting from T");
        List<Vertex<Character>> result = GraphAlgorithms.bfs(new Vertex<>('T'), largeGraph);
        assertEquals(expected, result); // Check amount & order of vertices returned
    }

    @Test(timeout = TIMEOUT)
    public void testBFSLarge_U() {
        // Test starting at vertex U
        List<Vertex<Character>> expected = new LinkedList<>();
        expected.add(new Vertex<>('U'));
        expected.add(new Vertex<>('J'));
        expected.add(new Vertex<>('P'));
        expected.add(new Vertex<>('A'));
        expected.add(new Vertex<>('N'));
        expected.add(new Vertex<>('T'));
        expected.add(new Vertex<>('W'));
        expected.add(new Vertex<>('B'));
        expected.add(new Vertex<>('C'));
        expected.add(new Vertex<>('V'));
        expected.add(new Vertex<>('S'));
        expected.add(new Vertex<>('Y'));
        expected.add(new Vertex<>('G'));
        expected.add(new Vertex<>('Z'));
        expected.add(new Vertex<>('L'));

        System.out.println("BFS starting from U");
        List<Vertex<Character>> result = GraphAlgorithms.bfs(new Vertex<>('U'), largeGraph);
        assertEquals(expected, result); // Check amount & order of vertices returned
    }

    @Test(timeout = TIMEOUT)
    public void testBFSLarge_V() {
        // Test starting at vertex V
        List<Vertex<Character>> expected = new LinkedList<>();
        expected.add(new Vertex<>('V'));
        expected.add(new Vertex<>('A'));
        expected.add(new Vertex<>('B'));
        expected.add(new Vertex<>('C'));
        expected.add(new Vertex<>('J'));
        expected.add(new Vertex<>('Y'));
        expected.add(new Vertex<>('Z'));
        expected.add(new Vertex<>('L'));
        expected.add(new Vertex<>('N'));
        expected.add(new Vertex<>('U'));
        expected.add(new Vertex<>('S'));
        expected.add(new Vertex<>('P'));
        expected.add(new Vertex<>('T'));
        expected.add(new Vertex<>('W'));
        expected.add(new Vertex<>('G'));

        System.out.println("BFS starting from V");
        List<Vertex<Character>> result = GraphAlgorithms.bfs(new Vertex<>('V'), largeGraph);
        assertEquals(expected, result); // Check amount & order of vertices returned
    }

    @Test(timeout = TIMEOUT)
    public void testBFSLarge_W() {
        // Test starting at vertex W
        List<Vertex<Character>> expected = new LinkedList<>();
        expected.add(new Vertex<>('W'));
        expected.add(new Vertex<>('G'));
        expected.add(new Vertex<>('P'));
        expected.add(new Vertex<>('T'));
        expected.add(new Vertex<>('U'));
        expected.add(new Vertex<>('J'));
        expected.add(new Vertex<>('A'));
        expected.add(new Vertex<>('N'));
        expected.add(new Vertex<>('B'));
        expected.add(new Vertex<>('C'));
        expected.add(new Vertex<>('V'));
        expected.add(new Vertex<>('S'));
        expected.add(new Vertex<>('Y'));
        expected.add(new Vertex<>('Z'));
        expected.add(new Vertex<>('L'));

        System.out.println("BFS starting from W");
        List<Vertex<Character>> result = GraphAlgorithms.bfs(new Vertex<>('W'), largeGraph);
        assertEquals(expected, result); // Check amount & order of vertices returned
    }

    @Test(timeout = TIMEOUT)
    public void testBFSLarge_X() {
        // Test starting at vertex X
        List<Vertex<Character>> expected = new LinkedList<>();
        expected.add(new Vertex<>('X'));
        expected.add(new Vertex<>('I'));
        expected.add(new Vertex<>('Q'));
        expected.add(new Vertex<>('O'));
        expected.add(new Vertex<>('D'));
        expected.add(new Vertex<>('F'));
        expected.add(new Vertex<>('M'));
        expected.add(new Vertex<>('H'));

        System.out.println("BFS starting from X");
        List<Vertex<Character>> result = GraphAlgorithms.bfs(new Vertex<>('X'), largeGraph);
        assertEquals(expected, result); // Check amount & order of vertices returned
    }

    @Test(timeout = TIMEOUT)
    public void testBFSLarge_Y() {
        // Test starting at vertex Y
        List<Vertex<Character>> expected = new LinkedList<>();
        expected.add(new Vertex<>('Y'));
        expected.add(new Vertex<>('B'));
        expected.add(new Vertex<>('N'));
        expected.add(new Vertex<>('Z'));
        expected.add(new Vertex<>('A'));
        expected.add(new Vertex<>('J'));
        expected.add(new Vertex<>('S'));
        expected.add(new Vertex<>('C'));
        expected.add(new Vertex<>('V'));
        expected.add(new Vertex<>('U'));
        expected.add(new Vertex<>('L'));
        expected.add(new Vertex<>('P'));
        expected.add(new Vertex<>('T'));
        expected.add(new Vertex<>('W'));
        expected.add(new Vertex<>('G'));

        System.out.println("BFS starting from Y");
        List<Vertex<Character>> result = GraphAlgorithms.bfs(new Vertex<>('Y'), largeGraph);
        assertEquals(expected, result); // Check amount & order of vertices returned
    }

    @Test(timeout = TIMEOUT)
    public void testBFSLarge_Z() {
        // Test starting at vertex Z
        List<Vertex<Character>> expected = new LinkedList<>();
        expected.add(new Vertex<>('Z'));
        expected.add(new Vertex<>('B'));
        expected.add(new Vertex<>('C'));
        expected.add(new Vertex<>('Y'));
        expected.add(new Vertex<>('A'));
        expected.add(new Vertex<>('L'));
        expected.add(new Vertex<>('N'));
        expected.add(new Vertex<>('J'));
        expected.add(new Vertex<>('V'));
        expected.add(new Vertex<>('S'));
        expected.add(new Vertex<>('U'));
        expected.add(new Vertex<>('P'));
        expected.add(new Vertex<>('T'));
        expected.add(new Vertex<>('W'));
        expected.add(new Vertex<>('G'));

        System.out.println("BFS starting from Z");
        List<Vertex<Character>> result = GraphAlgorithms.bfs(new Vertex<>('Z'), largeGraph);
        assertEquals(expected, result); // Check amount & order of vertices returned
    }

    @Test(timeout = TIMEOUT)
    public void testDFSLarge_A() {
        // Test starting at vertex A
        List<Vertex<Character>> expected = new LinkedList<>();
        expected.add(new Vertex<>('A'));
        expected.add(new Vertex<>('B'));
        expected.add(new Vertex<>('Y'));
        expected.add(new Vertex<>('N'));
        expected.add(new Vertex<>('J'));
        expected.add(new Vertex<>('U'));
        expected.add(new Vertex<>('P'));
        expected.add(new Vertex<>('T'));
        expected.add(new Vertex<>('G'));
        expected.add(new Vertex<>('W'));
        expected.add(new Vertex<>('S'));
        expected.add(new Vertex<>('Z'));
        expected.add(new Vertex<>('C'));
        expected.add(new Vertex<>('L'));
        expected.add(new Vertex<>('V'));

        System.out.println("DFS starting from A");
        List<Vertex<Character>> result = GraphAlgorithms.dfs(new Vertex<>('A'), largeGraph);
        assertEquals(expected, result); // Check amount & order of vertices returned
    }

    @Test(timeout = TIMEOUT)
    public void testDFSLarge_B() {
        // Test starting at vertex B
        List<Vertex<Character>> expected = new LinkedList<>();
        expected.add(new Vertex<>('B'));
        expected.add(new Vertex<>('A'));
        expected.add(new Vertex<>('C'));
        expected.add(new Vertex<>('L'));
        expected.add(new Vertex<>('Z'));
        expected.add(new Vertex<>('Y'));
        expected.add(new Vertex<>('N'));
        expected.add(new Vertex<>('J'));
        expected.add(new Vertex<>('U'));
        expected.add(new Vertex<>('P'));
        expected.add(new Vertex<>('T'));
        expected.add(new Vertex<>('G'));
        expected.add(new Vertex<>('W'));
        expected.add(new Vertex<>('S'));
        expected.add(new Vertex<>('V'));

        System.out.println("DFS starting from B");
        List<Vertex<Character>> result = GraphAlgorithms.dfs(new Vertex<>('B'), largeGraph);
        assertEquals(expected, result); // Check amount & order of vertices returned
    }

    @Test(timeout = TIMEOUT)
    public void testDFSLarge_C() {
        // Test starting at vertex C
        List<Vertex<Character>> expected = new LinkedList<>();
        expected.add(new Vertex<>('C'));
        expected.add(new Vertex<>('A'));
        expected.add(new Vertex<>('B'));
        expected.add(new Vertex<>('Y'));
        expected.add(new Vertex<>('N'));
        expected.add(new Vertex<>('J'));
        expected.add(new Vertex<>('U'));
        expected.add(new Vertex<>('P'));
        expected.add(new Vertex<>('T'));
        expected.add(new Vertex<>('G'));
        expected.add(new Vertex<>('W'));
        expected.add(new Vertex<>('S'));
        expected.add(new Vertex<>('Z'));
        expected.add(new Vertex<>('V'));
        expected.add(new Vertex<>('L'));

        System.out.println("DFS starting from C");
        List<Vertex<Character>> result = GraphAlgorithms.dfs(new Vertex<>('C'), largeGraph);
        assertEquals(expected, result); // Check amount & order of vertices returned
    }

    @Test(timeout = TIMEOUT)
    public void testDFSLarge_D() {
        // Test starting at vertex D
        List<Vertex<Character>> expected = new LinkedList<>();
        expected.add(new Vertex<>('D'));
        expected.add(new Vertex<>('M'));
        expected.add(new Vertex<>('F'));
        expected.add(new Vertex<>('O'));
        expected.add(new Vertex<>('Q'));
        expected.add(new Vertex<>('I'));
        expected.add(new Vertex<>('X'));
        expected.add(new Vertex<>('H'));

        System.out.println("DFS starting from D");
        List<Vertex<Character>> result = GraphAlgorithms.dfs(new Vertex<>('D'), largeGraph);
        assertEquals(expected, result); // Check amount & order of vertices returned
    }

    @Test(timeout = TIMEOUT)
    public void testDFSLarge_E() {
        // Test starting at vertex E
        List<Vertex<Character>> expected = new LinkedList<>();
        expected.add(new Vertex<>('E'));

        System.out.println("DFS starting from E");
        List<Vertex<Character>> result = GraphAlgorithms.dfs(new Vertex<>('E'), largeGraph);
        assertEquals(expected, result); // Check amount & order of vertices returned
    }

    @Test(timeout = TIMEOUT)
    public void testDFSLarge_F() {
        // Test starting at vertex F
        List<Vertex<Character>> expected = new LinkedList<>();
        expected.add(new Vertex<>('F'));
        expected.add(new Vertex<>('M'));
        expected.add(new Vertex<>('D'));
        expected.add(new Vertex<>('O'));
        expected.add(new Vertex<>('Q'));
        expected.add(new Vertex<>('I'));
        expected.add(new Vertex<>('X'));
        expected.add(new Vertex<>('H'));

        System.out.println("DFS starting from F");
        List<Vertex<Character>> result = GraphAlgorithms.dfs(new Vertex<>('F'), largeGraph);
        assertEquals(expected, result); // Check amount & order of vertices returned
    }

    @Test(timeout = TIMEOUT)
    public void testDFSLarge_G() {
        // Test starting at vertex G
        List<Vertex<Character>> expected = new LinkedList<>();
        expected.add(new Vertex<>('G'));
        expected.add(new Vertex<>('T'));
        expected.add(new Vertex<>('P'));
        expected.add(new Vertex<>('U'));
        expected.add(new Vertex<>('J'));
        expected.add(new Vertex<>('A'));
        expected.add(new Vertex<>('B'));
        expected.add(new Vertex<>('Y'));
        expected.add(new Vertex<>('N'));
        expected.add(new Vertex<>('S'));
        expected.add(new Vertex<>('Z'));
        expected.add(new Vertex<>('C'));
        expected.add(new Vertex<>('L'));
        expected.add(new Vertex<>('V'));
        expected.add(new Vertex<>('W'));

        System.out.println("DFS starting from G");
        List<Vertex<Character>> result = GraphAlgorithms.dfs(new Vertex<>('G'), largeGraph);
        assertEquals(expected, result); // Check amount & order of vertices returned
    }

    @Test(timeout = TIMEOUT)
    public void testDFSLarge_H() {
        // Test starting at vertex H
        List<Vertex<Character>> expected = new LinkedList<>();
        expected.add(new Vertex<>('H'));
        expected.add(new Vertex<>('M'));
        expected.add(new Vertex<>('D'));
        expected.add(new Vertex<>('O'));
        expected.add(new Vertex<>('F'));
        expected.add(new Vertex<>('Q'));
        expected.add(new Vertex<>('I'));
        expected.add(new Vertex<>('X'));

        System.out.println("DFS starting from H");
        List<Vertex<Character>> result = GraphAlgorithms.dfs(new Vertex<>('H'), largeGraph);
        assertEquals(expected, result); // Check amount & order of vertices returned
    }

    @Test(timeout = TIMEOUT)
    public void testDFSLarge_I() {
        // Test starting at vertex I
        List<Vertex<Character>> expected = new LinkedList<>();
        expected.add(new Vertex<>('I'));
        expected.add(new Vertex<>('Q'));
        expected.add(new Vertex<>('O'));
        expected.add(new Vertex<>('D'));
        expected.add(new Vertex<>('M'));
        expected.add(new Vertex<>('F'));
        expected.add(new Vertex<>('H'));
        expected.add(new Vertex<>('X'));

        System.out.println("DFS starting from I");
        List<Vertex<Character>> result = GraphAlgorithms.dfs(new Vertex<>('I'), largeGraph);
        assertEquals(expected, result); // Check amount & order of vertices returned
    }

    @Test(timeout = TIMEOUT)
    public void testDFSLarge_J() {
        // Test starting at vertex J
        List<Vertex<Character>> expected = new LinkedList<>();
        expected.add(new Vertex<>('J'));
        expected.add(new Vertex<>('A'));
        expected.add(new Vertex<>('B'));
        expected.add(new Vertex<>('Y'));
        expected.add(new Vertex<>('N'));
        expected.add(new Vertex<>('S'));
        expected.add(new Vertex<>('Z'));
        expected.add(new Vertex<>('C'));
        expected.add(new Vertex<>('L'));
        expected.add(new Vertex<>('V'));
        expected.add(new Vertex<>('U'));
        expected.add(new Vertex<>('P'));
        expected.add(new Vertex<>('T'));
        expected.add(new Vertex<>('G'));
        expected.add(new Vertex<>('W'));

        System.out.println("DFS starting from J");
        List<Vertex<Character>> result = GraphAlgorithms.dfs(new Vertex<>('J'), largeGraph);
        assertEquals(expected, result); // Check amount & order of vertices returned
    }

    @Test(timeout = TIMEOUT)
    public void testDFSLarge_K() {
        // Test starting at vertex K
        List<Vertex<Character>> expected = new LinkedList<>();
        expected.add(new Vertex<>('K'));
        expected.add(new Vertex<>('R'));

        System.out.println("DFS starting from K");
        List<Vertex<Character>> result = GraphAlgorithms.dfs(new Vertex<>('K'), largeGraph);
        assertEquals(expected, result); // Check amount & order of vertices returned
    }

    @Test(timeout = TIMEOUT)
    public void testDFSLarge_L() {
        // Test starting at vertex L
        List<Vertex<Character>> expected = new LinkedList<>();
        expected.add(new Vertex<>('L'));
        expected.add(new Vertex<>('C'));
        expected.add(new Vertex<>('A'));
        expected.add(new Vertex<>('B'));
        expected.add(new Vertex<>('Y'));
        expected.add(new Vertex<>('N'));
        expected.add(new Vertex<>('J'));
        expected.add(new Vertex<>('U'));
        expected.add(new Vertex<>('P'));
        expected.add(new Vertex<>('T'));
        expected.add(new Vertex<>('G'));
        expected.add(new Vertex<>('W'));
        expected.add(new Vertex<>('S'));
        expected.add(new Vertex<>('Z'));
        expected.add(new Vertex<>('V'));

        System.out.println("DFS starting from L");
        List<Vertex<Character>> result = GraphAlgorithms.dfs(new Vertex<>('L'), largeGraph);
        assertEquals(expected, result); // Check amount & order of vertices returned
    }

    @Test(timeout = TIMEOUT)
    public void testDFSLarge_M() {
        // Test starting at vertex M
        List<Vertex<Character>> expected = new LinkedList<>();
        expected.add(new Vertex<>('M'));
        expected.add(new Vertex<>('D'));
        expected.add(new Vertex<>('O'));
        expected.add(new Vertex<>('F'));
        expected.add(new Vertex<>('Q'));
        expected.add(new Vertex<>('I'));
        expected.add(new Vertex<>('X'));
        expected.add(new Vertex<>('H'));

        System.out.println("DFS starting from M");
        List<Vertex<Character>> result = GraphAlgorithms.dfs(new Vertex<>('M'), largeGraph);
        assertEquals(expected, result); // Check amount & order of vertices returned
    }

    @Test(timeout = TIMEOUT)
    public void testDFSLarge_N() {
        // Test starting at vertex N
        List<Vertex<Character>> expected = new LinkedList<>();
        expected.add(new Vertex<>('N'));
        expected.add(new Vertex<>('J'));
        expected.add(new Vertex<>('A'));
        expected.add(new Vertex<>('B'));
        expected.add(new Vertex<>('Y'));
        expected.add(new Vertex<>('Z'));
        expected.add(new Vertex<>('C'));
        expected.add(new Vertex<>('L'));
        expected.add(new Vertex<>('V'));
        expected.add(new Vertex<>('U'));
        expected.add(new Vertex<>('P'));
        expected.add(new Vertex<>('T'));
        expected.add(new Vertex<>('G'));
        expected.add(new Vertex<>('W'));
        expected.add(new Vertex<>('S'));

        System.out.println("DFS starting from N");
        List<Vertex<Character>> result = GraphAlgorithms.dfs(new Vertex<>('N'), largeGraph);
        assertEquals(expected, result); // Check amount & order of vertices returned
    }

    @Test(timeout = TIMEOUT)
    public void testDFSLarge_O() {
        // Test starting at vertex O
        List<Vertex<Character>> expected = new LinkedList<>();
        expected.add(new Vertex<>('O'));
        expected.add(new Vertex<>('D'));
        expected.add(new Vertex<>('M'));
        expected.add(new Vertex<>('F'));
        expected.add(new Vertex<>('H'));
        expected.add(new Vertex<>('Q'));
        expected.add(new Vertex<>('I'));
        expected.add(new Vertex<>('X'));

        System.out.println("DFS starting from O");
        List<Vertex<Character>> result = GraphAlgorithms.dfs(new Vertex<>('O'), largeGraph);
        assertEquals(expected, result); // Check amount & order of vertices returned
    }

    @Test(timeout = TIMEOUT)
    public void testDFSLarge_P() {
        // Test starting at vertex P
        List<Vertex<Character>> expected = new LinkedList<>();
        expected.add(new Vertex<>('P'));
        expected.add(new Vertex<>('T'));
        expected.add(new Vertex<>('G'));
        expected.add(new Vertex<>('W'));
        expected.add(new Vertex<>('U'));
        expected.add(new Vertex<>('J'));
        expected.add(new Vertex<>('A'));
        expected.add(new Vertex<>('B'));
        expected.add(new Vertex<>('Y'));
        expected.add(new Vertex<>('N'));
        expected.add(new Vertex<>('S'));
        expected.add(new Vertex<>('Z'));
        expected.add(new Vertex<>('C'));
        expected.add(new Vertex<>('L'));
        expected.add(new Vertex<>('V'));

        System.out.println("DFS starting from P");
        List<Vertex<Character>> result = GraphAlgorithms.dfs(new Vertex<>('P'), largeGraph);
        assertEquals(expected, result); // Check amount & order of vertices returned
    }

    @Test(timeout = TIMEOUT)
    public void testDFSLarge_Q() {
        // Test starting at vertex Q
        List<Vertex<Character>> expected = new LinkedList<>();
        expected.add(new Vertex<>('Q'));
        expected.add(new Vertex<>('I'));
        expected.add(new Vertex<>('X'));
        expected.add(new Vertex<>('O'));
        expected.add(new Vertex<>('D'));
        expected.add(new Vertex<>('M'));
        expected.add(new Vertex<>('F'));
        expected.add(new Vertex<>('H'));

        System.out.println("DFS starting from Q");
        List<Vertex<Character>> result = GraphAlgorithms.dfs(new Vertex<>('Q'), largeGraph);
        assertEquals(expected, result); // Check amount & order of vertices returned
    }

    @Test(timeout = TIMEOUT)
    public void testDFSLarge_R() {
        // Test starting at vertex R
        List<Vertex<Character>> expected = new LinkedList<>();
        expected.add(new Vertex<>('R'));
        expected.add(new Vertex<>('K'));

        System.out.println("DFS starting from R");
        List<Vertex<Character>> result = GraphAlgorithms.dfs(new Vertex<>('R'), largeGraph);
        assertEquals(expected, result); // Check amount & order of vertices returned
    }

    @Test(timeout = TIMEOUT)
    public void testDFSLarge_S() {
        // Test starting at vertex S
        List<Vertex<Character>> expected = new LinkedList<>();
        expected.add(new Vertex<>('S'));
        expected.add(new Vertex<>('N'));
        expected.add(new Vertex<>('J'));
        expected.add(new Vertex<>('A'));
        expected.add(new Vertex<>('B'));
        expected.add(new Vertex<>('Y'));
        expected.add(new Vertex<>('Z'));
        expected.add(new Vertex<>('C'));
        expected.add(new Vertex<>('L'));
        expected.add(new Vertex<>('V'));
        expected.add(new Vertex<>('U'));
        expected.add(new Vertex<>('P'));
        expected.add(new Vertex<>('T'));
        expected.add(new Vertex<>('G'));
        expected.add(new Vertex<>('W'));

        System.out.println("DFS starting from S");
        List<Vertex<Character>> result = GraphAlgorithms.dfs(new Vertex<>('S'), largeGraph);
        assertEquals(expected, result); // Check amount & order of vertices returned
    }

    @Test(timeout = TIMEOUT)
    public void testDFSLarge_T() {
        // Test starting at vertex T
        List<Vertex<Character>> expected = new LinkedList<>();
        expected.add(new Vertex<>('T'));
        expected.add(new Vertex<>('G'));
        expected.add(new Vertex<>('W'));
        expected.add(new Vertex<>('P'));
        expected.add(new Vertex<>('U'));
        expected.add(new Vertex<>('J'));
        expected.add(new Vertex<>('A'));
        expected.add(new Vertex<>('B'));
        expected.add(new Vertex<>('Y'));
        expected.add(new Vertex<>('N'));
        expected.add(new Vertex<>('S'));
        expected.add(new Vertex<>('Z'));
        expected.add(new Vertex<>('C'));
        expected.add(new Vertex<>('L'));
        expected.add(new Vertex<>('V'));

        System.out.println("DFS starting from T");
        List<Vertex<Character>> result = GraphAlgorithms.dfs(new Vertex<>('T'), largeGraph);
        assertEquals(expected, result); // Check amount & order of vertices returned
    }

    @Test(timeout = TIMEOUT)
    public void testDFSLarge_U() {
        // Test starting at vertex U
        List<Vertex<Character>> expected = new LinkedList<>();
        expected.add(new Vertex<>('U'));
        expected.add(new Vertex<>('J'));
        expected.add(new Vertex<>('A'));
        expected.add(new Vertex<>('B'));
        expected.add(new Vertex<>('Y'));
        expected.add(new Vertex<>('N'));
        expected.add(new Vertex<>('S'));
        expected.add(new Vertex<>('Z'));
        expected.add(new Vertex<>('C'));
        expected.add(new Vertex<>('L'));
        expected.add(new Vertex<>('V'));
        expected.add(new Vertex<>('P'));
        expected.add(new Vertex<>('T'));
        expected.add(new Vertex<>('G'));
        expected.add(new Vertex<>('W'));

        System.out.println("DFS starting from U");
        List<Vertex<Character>> result = GraphAlgorithms.dfs(new Vertex<>('U'), largeGraph);
        assertEquals(expected, result); // Check amount & order of vertices returned
    }

    @Test(timeout = TIMEOUT)
    public void testDFSLarge_V() {
        // Test starting at vertex V
        List<Vertex<Character>> expected = new LinkedList<>();
        expected.add(new Vertex<>('V'));
        expected.add(new Vertex<>('A'));
        expected.add(new Vertex<>('B'));
        expected.add(new Vertex<>('Y'));
        expected.add(new Vertex<>('N'));
        expected.add(new Vertex<>('J'));
        expected.add(new Vertex<>('U'));
        expected.add(new Vertex<>('P'));
        expected.add(new Vertex<>('T'));
        expected.add(new Vertex<>('G'));
        expected.add(new Vertex<>('W'));
        expected.add(new Vertex<>('S'));
        expected.add(new Vertex<>('Z'));
        expected.add(new Vertex<>('C'));
        expected.add(new Vertex<>('L'));

        System.out.println("DFS starting from V");
        List<Vertex<Character>> result = GraphAlgorithms.dfs(new Vertex<>('V'), largeGraph);
        assertEquals(expected, result); // Check amount & order of vertices returned
    }

    @Test(timeout = TIMEOUT)
    public void testDFSLarge_W() {
        // Test starting at vertex W
        List<Vertex<Character>> expected = new LinkedList<>();
        expected.add(new Vertex<>('W'));
        expected.add(new Vertex<>('G'));
        expected.add(new Vertex<>('T'));
        expected.add(new Vertex<>('P'));
        expected.add(new Vertex<>('U'));
        expected.add(new Vertex<>('J'));
        expected.add(new Vertex<>('A'));
        expected.add(new Vertex<>('B'));
        expected.add(new Vertex<>('Y'));
        expected.add(new Vertex<>('N'));
        expected.add(new Vertex<>('S'));
        expected.add(new Vertex<>('Z'));
        expected.add(new Vertex<>('C'));
        expected.add(new Vertex<>('L'));
        expected.add(new Vertex<>('V'));

        System.out.println("DFS starting from W");
        List<Vertex<Character>> result = GraphAlgorithms.dfs(new Vertex<>('W'), largeGraph);
        assertEquals(expected, result); // Check amount & order of vertices returned
    }

    @Test(timeout = TIMEOUT)
    public void testDFSLarge_X() {
        // Test starting at vertex X
        List<Vertex<Character>> expected = new LinkedList<>();
        expected.add(new Vertex<>('X'));
        expected.add(new Vertex<>('I'));
        expected.add(new Vertex<>('Q'));
        expected.add(new Vertex<>('O'));
        expected.add(new Vertex<>('D'));
        expected.add(new Vertex<>('M'));
        expected.add(new Vertex<>('F'));
        expected.add(new Vertex<>('H'));

        System.out.println("DFS starting from X");
        List<Vertex<Character>> result = GraphAlgorithms.dfs(new Vertex<>('X'), largeGraph);
        assertEquals(expected, result); // Check amount & order of vertices returned
    }

    @Test(timeout = TIMEOUT)
    public void testDFSLarge_Y() {
        // Test starting at vertex Y
        List<Vertex<Character>> expected = new LinkedList<>();
        expected.add(new Vertex<>('Y'));
        expected.add(new Vertex<>('B'));
        expected.add(new Vertex<>('A'));
        expected.add(new Vertex<>('C'));
        expected.add(new Vertex<>('L'));
        expected.add(new Vertex<>('Z'));
        expected.add(new Vertex<>('J'));
        expected.add(new Vertex<>('N'));
        expected.add(new Vertex<>('S'));
        expected.add(new Vertex<>('U'));
        expected.add(new Vertex<>('P'));
        expected.add(new Vertex<>('T'));
        expected.add(new Vertex<>('G'));
        expected.add(new Vertex<>('W'));
        expected.add(new Vertex<>('V'));

        System.out.println("DFS starting from Y");
        List<Vertex<Character>> result = GraphAlgorithms.dfs(new Vertex<>('Y'), largeGraph);
        assertEquals(expected, result); // Check amount & order of vertices returned
    }

    @Test(timeout = TIMEOUT)
    public void testDFSLarge_Z() {
        // Test starting at vertex Z
        List<Vertex<Character>> expected = new LinkedList<>();
        expected.add(new Vertex<>('Z'));
        expected.add(new Vertex<>('B'));
        expected.add(new Vertex<>('A'));
        expected.add(new Vertex<>('C'));
        expected.add(new Vertex<>('L'));
        expected.add(new Vertex<>('J'));
        expected.add(new Vertex<>('N'));
        expected.add(new Vertex<>('S'));
        expected.add(new Vertex<>('Y'));
        expected.add(new Vertex<>('U'));
        expected.add(new Vertex<>('P'));
        expected.add(new Vertex<>('T'));
        expected.add(new Vertex<>('G'));
        expected.add(new Vertex<>('W'));
        expected.add(new Vertex<>('V'));

        System.out.println("DFS starting from Z");
        List<Vertex<Character>> result = GraphAlgorithms.dfs(new Vertex<>('Z'), largeGraph);
        assertEquals(expected, result); // Check amount & order of vertices returned
    }
}
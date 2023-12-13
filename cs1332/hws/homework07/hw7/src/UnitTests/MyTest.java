import org.junit.Before;
import org.junit.Test;

import java.util.Arrays;
import java.util.List;
import java.util.ArrayList;
import java.util.NoSuchElementException;

import static org.junit.Assert.*;

public class MyTest {

    private static final int TIMEOUT = 200;
    private AVL<String> tree;
    private List<AVLNode<String>> list;

    private void preorder() {
        list = new ArrayList<>();
        preorderHelper(list, tree.getRoot());
    }

    private void preorderHelper(List<AVLNode<String>> list, AVLNode<String> node) {
        if (node != null) {
            list.add(node);
            preorderHelper(list, node.getLeft());
            preorderHelper(list, node.getRight());
        }
    }

    private String[] values() {
        String[] res = new String[list.size()];
        for (int i = 0; i < res.length; ++i) {
            res[i] = list.get(i).getData();
        }
        return res;
    }

    private int[] heights() {
        int[] res = new int[list.size()];
        for (int i = 0; i < res.length; ++i) {
            res[i] = list.get(i).getHeight();
        }
        return res;
    }

    private int[] bf() {
        int[] res = new int[list.size()];
        for (int i = 0; i < res.length; ++i) {
            res[i] = list.get(i).getBalanceFactor();
        }
        return res;
    }

    @Before
    public void setup() {
        tree = new AVL<>();
    }

    @Test(timeout = TIMEOUT)
    public void testInitialization() {
        assertEquals(0, tree.size());
        assertNull(tree.getRoot());
    }

    @Test(timeout = TIMEOUT)
    public void testAdd1() {
        tree = new AVL<>(Arrays.asList("020", "004"));
        preorder();
        String[] values = values();
        int[] heights = heights();
        int[] bf = bf();
        assertEquals(2, tree.size());
        assertArrayEquals(new String[] {"020", "004"}, values);
        assertArrayEquals(new int[] {1, 0}, heights);
        assertArrayEquals(new int[] {1, 0}, bf);

        tree.add("015");
        preorder();
        values = values();
        heights = heights();
        bf = bf();
        assertEquals(3, tree.size());
        assertArrayEquals(new String[] {"015", "004", "020"}, values);
        assertArrayEquals(new int[] {1, 0, 0}, heights);
        assertArrayEquals(new int[] {0, 0, 0}, bf);
    }

    @Test(timeout = TIMEOUT)
    public void testAdd2() {
        tree = new AVL<>(Arrays.asList("020", "004", "026", "003", "009"));
        preorder();
        String[] values = values();
        int[] heights = heights();
        int[] bf = bf();
        assertEquals(5, tree.size());
        assertArrayEquals(new String[] {"020", "004", "003", "009", "026"}, values);
        assertArrayEquals(new int[] {2, 1, 0, 0, 0}, heights);
        assertArrayEquals(new int[] {1, 0, 0, 0, 0}, bf);

        tree.add("015");
        preorder();
        values = values();
        heights = heights();
        bf = bf();
        assertEquals(6, tree.size());
        assertArrayEquals(new String[] {"009", "004", "003", "020", "015", "026"}, values);
        assertArrayEquals(new int[] {2, 1, 0, 1, 0, 0}, heights);
        assertArrayEquals(new int[] {0, 1, 0, 0, 0, 0}, bf);
    }

    @Test(timeout = TIMEOUT)
    public void testAdd3() {
        tree = new AVL<>(Arrays.asList("020", "004", "026", "003", "009", "021", "030", "002", "007",
                "011"));
        preorder();
        String[] values = values();
        int[] heights = heights();
        int[] bf = bf();
        assertEquals(10, tree.size());
        assertArrayEquals(new String[] {"020", "004", "003", "002", "009", "007", "011",
            "026", "021", "030"}, values);
        assertArrayEquals(new int[] {3, 2, 1, 0, 1, 0, 0, 1, 0, 0}, heights);
        assertArrayEquals(new int[] {1, 0, 1, 0, 0, 0, 0, 0, 0, 0}, bf);

        tree.add("015");
        preorder();
        values = values();
        heights = heights();
        bf = bf();
        assertEquals(11, tree.size());
        assertArrayEquals(new String[] {"009", "004", "003", "002", "007", "020", "011",
            "015", "026", "021", "030"}, values);
        assertArrayEquals(new int[] {3, 2, 1, 0, 0, 2, 1, 0, 1, 0, 0}, heights);
        assertArrayEquals(new int[] {0, 1, 1, 0, 0, 0, -1, 0, 0, 0, 0}, bf);
    }

    @Test(timeout = TIMEOUT)
    public void testAdd4() {
        tree = new AVL<>(Arrays.asList("020", "004", "026", "003", "009"));
        preorder();
        String[] values = values();
        int[] heights = heights();
        int[] bf = bf();
        assertEquals(5, tree.size());
        assertArrayEquals(new String[] {"020", "004", "003", "009", "026"}, values);
        assertArrayEquals(new int[] {2, 1, 0, 0, 0}, heights);
        assertArrayEquals(new int[] {1, 0, 0, 0, 0}, bf);

        tree.add("008");
        preorder();
        values = values();
        heights = heights();
        bf = bf();
        assertEquals(6, tree.size());
        assertArrayEquals(new String[] {"009", "004", "003", "008", "020", "026"}, values);
        assertArrayEquals(new int[] {2, 1, 0, 0, 1, 0}, heights);
        assertArrayEquals(new int[] {0, 0, 0, 0, -1, 0}, bf);
    }

    @Test(timeout = TIMEOUT)
    public void testAdd5() {
        tree = new AVL<>(Arrays.asList("020", "004", "026", "003", "009", "021", "030", "002", "007",
                "011"));
        preorder();
        String[] values = values();
        int[] heights = heights();
        int[] bf = bf();
        assertEquals(10, tree.size());
        assertArrayEquals(new String[] {"020", "004", "003", "002", "009", "007", "011",
                "026", "021", "030"}, values);
        assertArrayEquals(new int[] {3, 2, 1, 0, 1, 0, 0, 1, 0, 0}, heights);
        assertArrayEquals(new int[] {1, 0, 1, 0, 0, 0, 0, 0, 0, 0}, bf);

        tree.add("008");
        preorder();
        values = values();
        heights = heights();
        bf = bf();
        assertEquals(11, tree.size());
        assertArrayEquals(new String[] {"009", "004", "003", "002", "007", "008", "020",
            "011", "026", "021", "030"}, values);
        assertArrayEquals(new int[] {3, 2, 1, 0, 1, 0, 2, 0, 1, 0, 0}, heights);
        assertArrayEquals(new int[] {0, 0, 1, 0, -1, 0, -1, 0, 0, 0, 0}, bf);
    }

    @Test(timeout = TIMEOUT)
    public void testAdd6() {
        tree = new AVL<>(Arrays.asList("004", "005"));
        preorder();
        String[] values = values();
        int[] heights = heights();
        int[] bf = bf();
        assertEquals(2, tree.size());
        assertArrayEquals(new String[] {"004", "005"}, values);
        assertArrayEquals(new int[] {1, 0}, heights);
        assertArrayEquals(new int[] {-1, 0}, bf);

        tree.add("010");
        preorder();
        values = values();
        heights = heights();
        bf = bf();
        assertEquals(3, tree.size());
        assertArrayEquals(new String[] {"005", "004", "010"}, values);
        assertArrayEquals(new int[] {1, 0, 0}, heights);
        assertArrayEquals(new int[] {0, 0, 0}, bf);
    }

    @Test(timeout = TIMEOUT)
    public void testAdd7() {
        tree = new AVL<>(Arrays.asList("020", "004", "026", "021", "030"));
        preorder();
        String[] values = values();
        int[] heights = heights();
        int[] bf = bf();
        assertEquals(5, tree.size());
        assertArrayEquals(new String[] {"020", "004", "026", "021", "030"}, values);
        assertArrayEquals(new int[] {2, 0, 1, 0, 0}, heights);
        assertArrayEquals(new int[] {-1, 0, 0, 0, 0}, bf);

        tree.add("024");
        preorder();
        values = values();
        heights = heights();
        bf = bf();
        assertEquals(6, tree.size());
        assertArrayEquals(new String[] {"021", "020", "004", "026", "024", "030"}, values);
        assertArrayEquals(new int[] {2, 1, 0, 1, 0, 0}, heights);
        assertArrayEquals(new int[] {0, 1, 0, 0, 0, 0}, bf);
    }

    @Test(timeout = TIMEOUT)
    public void testAdd8() {
        tree = new AVL<>(Arrays.asList("020", "004", "026", "022", "030"));
        preorder();
        String[] values = values();
        int[] heights = heights();
        int[] bf = bf();
        assertEquals(5, tree.size());
        assertArrayEquals(new String[] {"020", "004", "026", "022", "030"}, values);
        assertArrayEquals(new int[] {2, 0, 1, 0, 0}, heights);
        assertArrayEquals(new int[] {-1, 0, 0, 0, 0}, bf);

        tree.add("021");
        preorder();
        values = values();
        heights = heights();
        bf = bf();
        assertEquals(6, tree.size());
        assertArrayEquals(new String[] {"022", "020", "004", "021", "026", "030"}, values);
        assertArrayEquals(new int[] {2, 1, 0, 0, 1, 0}, heights);
        assertArrayEquals(new int[] {0, 0, 0, 0, -1, 0}, bf);
    }

    @Test(timeout = TIMEOUT)
    public void testRemove1() {
        tree = new AVL<>(Arrays.asList("010", "015", "002", "001", "012", "017", "019"));
        preorder();
        String[] values = values();
        int[] heights = heights();
        int[] bf = bf();
        assertEquals(7, tree.size());
        assertArrayEquals(new String[] {"010", "002", "001", "015", "012", "017", "019"}, values);
        assertArrayEquals(new int[] {3, 1, 0, 2, 0, 1, 0}, heights);
        assertArrayEquals(new int[] {-1, 1, 0, -1, 0, -1, 0}, bf);

        assertEquals(new String("012"), tree.remove("012"));
        preorder();
        values = values();
        heights = heights();
        bf = bf();
        assertEquals(6, tree.size());
        assertArrayEquals(new String[] {"010", "002", "001", "017", "015", "019"}, values);
        assertArrayEquals(new int[] {2, 1, 0, 1, 0, 0}, heights);
        assertArrayEquals(new int[] {0, 1, 0, 0, 0, 0}, bf);
    }

    @Test(timeout = TIMEOUT)
    public void testRemove2() {
        tree = new AVL<>(Arrays.asList("010", "015", "002", "001", "012", "017", "019"));
        preorder();
        String[] values = values();
        int[] heights = heights();
        int[] bf = bf();
        assertEquals(7, tree.size());
        assertArrayEquals(new String[] {"010", "002", "001", "015", "012", "017", "019"}, values);
        assertArrayEquals(new int[] {3, 1, 0, 2, 0, 1, 0}, heights);
        assertArrayEquals(new int[] {-1, 1, 0, -1, 0, -1, 0}, bf);

        assertEquals(new String("001"), tree.remove("001"));
        preorder();
        values = values();
        heights = heights();
        bf = bf();
        assertEquals(6, tree.size());
        assertArrayEquals(new String[] {"015", "010", "002", "012", "017", "019"}, values);
        assertArrayEquals(new int[] {2, 1, 0, 0, 1, 0}, heights);
        assertArrayEquals(new int[] {0, 0, 0, 0, -1, 0}, bf);
    }

    @Test(timeout = TIMEOUT)
    public void testRemove3() {
        tree = new AVL<>(Arrays.asList("010", "015", "002", "001", "012", "017", "019"));
        preorder();
        String[] values = values();
        int[] heights = heights();
        int[] bf = bf();
        assertEquals(7, tree.size());
        assertArrayEquals(new String[] {"010", "002", "001", "015", "012", "017", "019"}, values);
        assertArrayEquals(new int[] {3, 1, 0, 2, 0, 1, 0}, heights);
        assertArrayEquals(new int[] {-1, 1, 0, -1, 0, -1, 0}, bf);

        assertEquals(new String("002"), tree.remove("002"));
        preorder();
        values = values();
        heights = heights();
        bf = bf();
        assertEquals(6, tree.size());
        assertArrayEquals(new String[] {"015", "010", "001", "012", "017", "019"}, values);
        assertArrayEquals(new int[] {2, 1, 0, 0, 1, 0}, heights);
        assertArrayEquals(new int[] {0, 0, 0, 0, -1, 0}, bf);
    }

    @Test(timeout = TIMEOUT)
    public void testRemove4() {
        tree = new AVL<>(Arrays.asList("010", "015", "002", "004", "012", "017", "019"));
        preorder();
        String[] values = values();
        int[] heights = heights();
        int[] bf = bf();
        assertEquals(7, tree.size());
        assertArrayEquals(new String[] {"010", "002", "004", "015", "012", "017", "019"}, values);
        assertArrayEquals(new int[] {3, 1, 0, 2, 0, 1, 0}, heights);
        assertArrayEquals(new int[] {-1, -1, 0, -1, 0, -1, 0}, bf);

        assertEquals(new String("002"), tree.remove("002"));
        preorder();
        values = values();
        heights = heights();
        bf = bf();
        assertEquals(6, tree.size());
        assertArrayEquals(new String[] {"015", "010", "004", "012", "017", "019"}, values);
        assertArrayEquals(new int[] {2, 1, 0, 0, 1, 0}, heights);
        assertArrayEquals(new int[] {0, 0, 0, 0, -1, 0}, bf);
    }

    @Test(timeout = TIMEOUT)
    public void testRemove5() {
        tree = new AVL<>(Arrays.asList("010", "015", "002", "001", "012", "017", "019"));
        preorder();
        String[] values = values();
        int[] heights = heights();
        int[] bf = bf();
        assertEquals(7, tree.size());
        assertArrayEquals(new String[] {"010", "002", "001", "015", "012", "017", "019"}, values);
        assertArrayEquals(new int[] {3, 1, 0, 2, 0, 1, 0}, heights);
        assertArrayEquals(new int[] {-1, 1, 0, -1, 0, -1, 0}, bf);

        assertEquals(new String("010"), tree.remove("010"));
        preorder();
        values = values();
        heights = heights();
        bf = bf();
        assertEquals(6, tree.size());
        assertArrayEquals(new String[] {"015", "002", "001", "012", "017", "019"}, values);
        assertArrayEquals(new int[] {2, 1, 0, 0, 1, 0}, heights);
        assertArrayEquals(new int[] {0, 0, 0, 0, -1, 0}, bf);
    }

    @Test(timeout = TIMEOUT)
    public void testRemove6() {
        tree = new AVL<>(Arrays.asList("010", "015", "002", "004", "012", "017", "019"));
        preorder();
        String[] values = values();
        int[] heights = heights();
        int[] bf = bf();
        assertEquals(7, tree.size());
        assertArrayEquals(new String[] {"010", "002", "004", "015", "012", "017", "019"}, values);
        assertArrayEquals(new int[] {3, 1, 0, 2, 0, 1, 0}, heights);
        assertArrayEquals(new int[] {-1, -1, 0, -1, 0, -1, 0}, bf);

        assertEquals(new String("010"), tree.remove("010"));
        preorder();
        values = values();
        heights = heights();
        bf = bf();
        assertEquals(6, tree.size());
        assertArrayEquals(new String[] {"015", "004", "002", "012", "017", "019"}, values);
        assertArrayEquals(new int[] {2, 1, 0, 0, 1, 0}, heights);
        assertArrayEquals(new int[] {0, 0, 0, 0, -1, 0}, bf);
    }

    @Test(timeout = TIMEOUT)
    public void testRemove7() {
        tree = new AVL<>(Arrays.asList("005", "002", "008", "001", "003", "007", "010",
                "004", "006", "009", "011", "012"));
        preorder();
        String[] values = values();
        int[] heights = heights();
        int[] bf = bf();
        assertEquals(12, tree.size());
        assertArrayEquals(new String[] {"005", "002", "001", "003", "004", "008", "007",
            "006", "010", "009", "011", "012"}, values);
        assertArrayEquals(new int[] {4, 2, 0, 1, 0, 3, 1, 0, 2, 0, 1, 0}, heights);
        assertArrayEquals(new int[] {-1, -1, 0, -1, 0, -1, 1, 0, -1, 0, -1, 0}, bf);

        assertEquals(new String("001"), tree.remove("001"));
        preorder();
        values = values();
        heights = heights();
        bf = bf();
        assertEquals(11, tree.size());
        assertArrayEquals(new String[] {"008", "005", "003", "002", "004", "007", "006",
            "010", "009", "011", "012"}, values);
        assertArrayEquals(new int[] {3, 2, 1, 0, 0, 1, 0, 2, 0, 1, 0}, heights);
        assertArrayEquals(new int[] {0, 0, 0, 0, 0, 1, 0, -1, 0, -1, 0}, bf);
    }

    @Test(timeout = TIMEOUT)
    public void testRemove8() {
        tree = new AVL<>(Arrays.asList("005", "002", "008", "001", "003", "007", "010",
                "004", "006", "009", "011", "012"));
        preorder();
        String[] values = values();
        int[] heights = heights();
        int[] bf = bf();
        assertEquals(12, tree.size());
        assertArrayEquals(new String[] {"005", "002", "001", "003", "004", "008", "007",
                "006", "010", "009", "011", "012"}, values);
        assertArrayEquals(new int[] {4, 2, 0, 1, 0, 3, 1, 0, 2, 0, 1, 0}, heights);
        assertArrayEquals(new int[] {-1, -1, 0, -1, 0, -1, 1, 0, -1, 0, -1, 0}, bf);

        assertEquals(new String("005"), tree.remove("005"));
        preorder();
        values = values();
        heights = heights();
        bf = bf();
        assertEquals(11, tree.size());
        assertArrayEquals(new String[] {"008", "004", "002", "001", "003", "007", "006", "010", "009",
            "011", "012"}, values);
        assertArrayEquals(new int[] {3, 2, 1, 0, 0, 1, 0, 2, 0, 1, 0}, heights);
        assertArrayEquals(new int[] {0, 0, 0, 0, 0, 1, 0, -1, 0, -1, 0}, bf);
    }

    @Test(timeout = TIMEOUT)
    public void testGet() {
        tree = new AVL<>(Arrays.asList("005", "002", "008", "001", "003", "007", "010",
                "004", "006", "009", "011", "012"));
        preorder();
        String[] values = values();
        int[] heights = heights();
        int[] bf = bf();
        assertEquals(12, tree.size());
        assertArrayEquals(new String[] {"005", "002", "001", "003", "004", "008", "007",
                "006", "010", "009", "011", "012"}, values);
        assertArrayEquals(new int[] {4, 2, 0, 1, 0, 3, 1, 0, 2, 0, 1, 0}, heights);
        assertArrayEquals(new int[] {-1, -1, 0, -1, 0, -1, 1, 0, -1, 0, -1, 0}, bf);

        String[] allElements = new String[] {"005", "002", "008", "001", "003", "007", "010",
                "004", "006", "009", "011", "012"};
        for (String str : allElements) {
            String s = new String(str);
            String res = tree.get(s);
            assertFalse(res == s);
            assertEquals(str, res);
        }
    }

    @Test(timeout = TIMEOUT)
    public void testContains() {
        tree = new AVL<>(Arrays.asList("005", "002", "008", "001", "003", "007", "010",
                "004", "006", "009", "011", "012"));
        preorder();
        String[] values = values();
        int[] heights = heights();
        int[] bf = bf();
        assertEquals(12, tree.size());
        assertArrayEquals(new String[] {"005", "002", "001", "003", "004", "008", "007",
                "006", "010", "009", "011", "012"}, values);
        assertArrayEquals(new int[] {4, 2, 0, 1, 0, 3, 1, 0, 2, 0, 1, 0}, heights);
        assertArrayEquals(new int[] {-1, -1, 0, -1, 0, -1, 1, 0, -1, 0, -1, 0}, bf);

        String[] allElements = new String[] {"005", "002", "008", "001", "003", "007", "010",
                "004", "006", "009", "011", "012"};
        for (String str : allElements) {
            String s = new String(str);
            assertTrue(tree.contains(s));
        }

        String[] notPartOf = new String[] {"0005", "5", "11", "013", "000", "014"};
        for (String str : notPartOf) {
            String s = new String(str);
            assertFalse(tree.contains(s));
        }
    }

    @Test(timeout = TIMEOUT)
    public void testHeight() {
        // The height should have been already tested lots of times in the previous tests
        // So I just include one special case here
        assertEquals(-1, tree.height());
    }

    @Test(timeout = TIMEOUT)
    public void testClear() {
        tree = new AVL<>(Arrays.asList("005", "002", "008", "001", "003", "007", "010",
                "004", "006", "009", "011", "012"));
        tree.clear();
        assertNull(tree.getRoot());
        assertEquals(0, tree.size());
    }

    @Test(timeout = TIMEOUT)
    public void testDeepestBranches1() {
        /*
                                10
                            /        \
                           5          15
                         /   \      /    \
                        2     7    13    20
                       / \   / \     \  / \
                      1   4 6   8   14 17  25
                     /           \          \
                    0             9         30
         */
        tree = new AVL<>(Arrays.asList("010", "005", "015", "002", "007", "013", "020",
                "001", "004", "006", "008", "014", "017", "025", "000", "009", "030"));
        List<String> expected = Arrays.asList("010", "005", "002", "001", "000", "007", "008",
                "009", "015", "020", "025", "030");
        List<String> actual = tree.deepestBranches();
        assertEquals(expected, actual);
    }

    @Test(timeout = TIMEOUT)
    public void testDeepestBranches2() {
        /*
                                10
                            /        \
                           5          15
                         /   \      /    \
                        2     7    13    20
                       / \   / \     \  / \
                      1   4 6   8   14 17  25
         */
        tree = new AVL<>(Arrays.asList("010", "005", "015", "002", "007", "013", "020",
                "001", "004", "006", "008", "014", "017", "025"));
        List<String> expected = Arrays.asList("010", "005", "002", "001", "004", "007",
                "006", "008", "015", "013", "014", "020", "017", "025");
        List<String> actual = tree.deepestBranches();
        assertEquals(expected, actual);
    }

    @Test(timeout = TIMEOUT)
    public void testDeepestBranches3() {
        /*
                                10
                            /        \
                           5          15
                         /   \      /
                        4     7    13
                             / \
                            6   8
         */
        tree = new AVL<>(Arrays.asList("010", "005", "015", "004", "007", "013", "006", "008"));
        List<String> expected = Arrays.asList("010", "005", "007", "006", "008");
        List<String> actual = tree.deepestBranches();
        assertEquals(expected, actual);
    }

    @Test(timeout = TIMEOUT)
    public void testDeepestBranches4() {
        /*
                                10
                            /        \
                           5          15
                         /   \      /    \
                        2     7    13    20
                               \     \
                                8    14
         */
        tree = new AVL<>(Arrays.asList("010", "005", "015", "002", "007", "013", "020",
                "008", "014"));
        List<String> expected = Arrays.asList("010", "005", "007", "008", "015", "013", "014");
        List<String> actual = tree.deepestBranches();
        assertEquals(expected, actual);
    }

    @Test(timeout = TIMEOUT)
    public void testDeepestBranches5() {
        /*
                                10
         */
        tree = new AVL<>(Arrays.asList("010"));
        List<String> expected = Arrays.asList("010");
        List<String> actual = tree.deepestBranches();
        assertEquals(expected, actual);

        tree.clear();
        expected = new ArrayList<>();
        actual = tree.deepestBranches();
        assertEquals(expected, actual);
    }

    @Test(timeout = TIMEOUT)
    public void testDeepestBranches6() {
        /*
                                10
                            /        \
                           5          15
                         /   \      /
                        4     7    13
                             /
                            6
         */
        tree = new AVL<>(Arrays.asList("010", "005", "015", "004", "007", "013", "006"));
        List<String> expected = Arrays.asList("010", "005", "007", "006");
        List<String> actual = tree.deepestBranches();
        assertEquals(expected, actual);
    }

    @Test(timeout = TIMEOUT)
    public void testSortedInBetween() {
        /*
                                10
                            /        \
                           5          15
                         /   \      /    \
                        2     7    13    20
                       / \   / \     \  / \
                      1   4 6   8   14 17  25
                     /           \          \
                    0             9         30
         */
        tree = new AVL<>(Arrays.asList("010", "005", "015", "002", "007", "013", "020",
                "001", "004", "006", "008", "014", "017", "025", "000", "009", "030"));
        List<String> expected = Arrays.asList("008", "009", "010", "013");
        List<String> actual = tree.sortedInBetween("007", "014");
        assertEquals(expected, actual);

        expected = Arrays.asList("004", "005", "006", "007");
        actual = tree.sortedInBetween("003", "008");
        assertEquals(expected, actual);

        expected = Arrays.asList();
        actual = tree.sortedInBetween("008", "008");
        assertEquals(expected, actual);

        expected = Arrays.asList("001", "002");
        actual = tree.sortedInBetween("000", "003");
        assertEquals(expected, actual);

        expected = Arrays.asList("015", "017", "020", "025");
        actual = tree.sortedInBetween("014", "029");
        assertEquals(expected, actual);

        expected = Arrays.asList("001", "002", "004", "005", "006", "007", "008", "009", "010",
                "013", "014", "015", "017", "020", "025", "030");
        actual = tree.sortedInBetween("000", "031");
        assertEquals(expected, actual);
    }

    @Test(timeout = TIMEOUT)
    public void testConstructorException() {
        assertThrows(IllegalArgumentException.class, () -> {
            tree = new AVL<>(null);
        });
        assertThrows(IllegalArgumentException.class, () -> {
            tree = new AVL<>(Arrays.asList("001", "002", null, "003"));
        });
    }

    @Test(timeout = TIMEOUT)
    public void testAddException() {
        assertThrows(IllegalArgumentException.class, () -> {
            tree.add(null);
        });
    }

    @Test(timeout = TIMEOUT)
    public void testRemoveException() {
        assertThrows(IllegalArgumentException.class, () -> {
            tree.remove(null);
        });
        tree = new AVL<>(Arrays.asList("005", "002", "008", "001", "003", "007", "010",
                "004", "006", "009", "011", "012"));
        String[] notPartOf = new String[] {"0005", "5", "11", "013", "000", "014"};
        for (String str : notPartOf) {
            assertThrows(NoSuchElementException.class, () -> {
                tree.remove(str);
            });
        }
    }

    @Test(timeout = TIMEOUT)
    public void testGetException() {
        assertThrows(IllegalArgumentException.class, () -> {
            tree.get(null);
        });
        tree = new AVL<>(Arrays.asList("005", "002", "008", "001", "003", "007", "010",
                "004", "006", "009", "011", "012"));
        String[] notPartOf = new String[] {"0005", "5", "11", "013", "000", "014"};
        for (String str : notPartOf) {
            assertThrows(NoSuchElementException.class, () -> {
                tree.get(str);
            });
        }
    }

    @Test(timeout = TIMEOUT)
    public void testContainsException() {
        assertThrows(IllegalArgumentException.class, () -> {
            tree.contains(null);
        });
    }

    @Test(timeout = TIMEOUT)
    public void testSortedInBetweenException() {
        assertThrows(IllegalArgumentException.class, () -> {
            tree.sortedInBetween(null, "001");
        });
        assertThrows(IllegalArgumentException.class, () -> {
            tree.sortedInBetween("null", null);
        });
    }
}
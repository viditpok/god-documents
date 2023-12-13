import org.junit.Before;
import org.junit.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.NoSuchElementException;

import static org.junit.Assert.*;

/**
 * @author Ahaan Limaye
 * @version 1.0
 */
public class AVLTest {
    private static final int TIMEOUT = 200;
    private AVL<Integer> tree;

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
    public void testConstructor() {
        List<Integer> list = new ArrayList<>();
        list.add(5);
        list.add(3);
        list.add(7);
        tree = new AVL<>(list);

        assertEquals(3, tree.size());

        assertEquals((Integer) 5, tree.getRoot().getData());
        assertEquals(1, tree.getRoot().getHeight());
        assertEquals(0, tree.getRoot().getBalanceFactor());
        assertEquals((Integer) 3, tree.getRoot().getLeft().getData());
        assertEquals(0, tree.getRoot().getLeft().getHeight());
        assertEquals(0, tree.getRoot().getLeft().getBalanceFactor());
        assertEquals((Integer) 7, tree.getRoot().getRight().getData());
        assertEquals(0, tree.getRoot().getRight().getHeight());
        assertEquals(0, tree.getRoot().getRight().getBalanceFactor());
    }

    @Test(timeout = TIMEOUT)
    public void testAdd() {
        /*
              5
             / \
            3   7
        */

        tree.add(5);
        tree.add(3);
        tree.add(7);

        assertEquals(3, tree.size());

        assertEquals((Integer) 5, tree.getRoot().getData());
        assertEquals(1, tree.getRoot().getHeight());
        assertEquals(0, tree.getRoot().getBalanceFactor());
        assertEquals((Integer) 3, tree.getRoot().getLeft().getData());
        assertEquals(0, tree.getRoot().getLeft().getHeight());
        assertEquals(0, tree.getRoot().getLeft().getBalanceFactor());
        assertEquals((Integer) 7, tree.getRoot().getRight().getData());
        assertEquals(0, tree.getRoot().getRight().getHeight());
        assertEquals(0, tree.getRoot().getRight().getBalanceFactor());
    }

    @Test(timeout = TIMEOUT)
    public void testAddDuplicate() {
        /*
              5
             / \
            3   7
        */

        tree.add(5);
        tree.add(3);
        tree.add(7);

        // duplicates
        tree.add(5);
        tree.add(3);
        tree.add(7);

        assertEquals(3, tree.size());

        assertEquals((Integer) 5, tree.getRoot().getData());
        assertEquals(1, tree.getRoot().getHeight());
        assertEquals(0, tree.getRoot().getBalanceFactor());
        assertEquals((Integer) 3, tree.getRoot().getLeft().getData());
        assertEquals(0, tree.getRoot().getLeft().getHeight());
        assertEquals(0, tree.getRoot().getLeft().getBalanceFactor());
        assertEquals((Integer) 7, tree.getRoot().getRight().getData());
        assertEquals(0, tree.getRoot().getRight().getHeight());
        assertEquals(0, tree.getRoot().getRight().getBalanceFactor());
    }

    @Test(timeout = TIMEOUT)
    public void testAddRighRightRotate() {
        /*
                  5
                 / \
                3   7
               /
              2
             /
            1
        */
        /*
                  5
                 / \
                2   7
               / \
              1   3
        */

        tree.add(5);
        tree.add(3);
        tree.add(7);
        tree.add(2);
        tree.add(1);

        assertEquals(5, tree.size());

        assertEquals((Integer) 5, tree.getRoot().getData());
        assertEquals(2, tree.getRoot().getHeight());
        assertEquals(1, tree.getRoot().getBalanceFactor());
        assertEquals((Integer) 2, tree.getRoot().getLeft().getData());
        assertEquals(1, tree.getRoot().getLeft().getHeight());
        assertEquals(0, tree.getRoot().getLeft().getBalanceFactor());
        assertEquals((Integer) 7, tree.getRoot().getRight().getData());
        assertEquals(0, tree.getRoot().getRight().getHeight());
        assertEquals(0, tree.getRoot().getRight().getBalanceFactor());
        assertEquals((Integer) 1, tree.getRoot().getLeft().getLeft().getData());
        assertEquals(0, tree.getRoot().getLeft().getLeft().getHeight());
        assertEquals(0, tree.getRoot().getLeft().getLeft().getBalanceFactor());
        assertEquals((Integer) 3, tree.getRoot().getLeft().getRight().getData());
        assertEquals(0, tree.getRoot().getLeft().getRight().getHeight());
        assertEquals(0, tree.getRoot().getLeft().getRight().getBalanceFactor());
    }

    @Test(timeout = TIMEOUT)
    public void testAddRightLeftRotate() {
        /*
                  5
                 / \
                3   7
               /
              1
               \
                2
        */
        /*
                  5
                 / \
                2   7
               / \
              1   3
        */

        tree.add(5);
        tree.add(3);
        tree.add(7);
        tree.add(1);
        tree.add(2);

        assertEquals(5, tree.size());

        assertEquals((Integer) 5, tree.getRoot().getData());
        assertEquals(2, tree.getRoot().getHeight());
        assertEquals(1, tree.getRoot().getBalanceFactor());
        assertEquals((Integer) 2, tree.getRoot().getLeft().getData());
        assertEquals(1, tree.getRoot().getLeft().getHeight());
        assertEquals(0, tree.getRoot().getLeft().getBalanceFactor());
        assertEquals((Integer) 7, tree.getRoot().getRight().getData());
        assertEquals(0, tree.getRoot().getRight().getHeight());
        assertEquals(0, tree.getRoot().getRight().getBalanceFactor());
        assertEquals((Integer) 1, tree.getRoot().getLeft().getLeft().getData());
        assertEquals(0, tree.getRoot().getLeft().getLeft().getHeight());
        assertEquals(0, tree.getRoot().getLeft().getLeft().getBalanceFactor());
        assertEquals((Integer) 3, tree.getRoot().getLeft().getRight().getData());
        assertEquals(0, tree.getRoot().getLeft().getRight().getHeight());
        assertEquals(0, tree.getRoot().getLeft().getRight().getBalanceFactor());
    }

    @Test(timeout = TIMEOUT)
    public void testAddLeftLeftRotate() {
        /*
                  5
                 / \
                3   7
                     \
                      8
                       \
                        9
        */
        /*
                  5
                 / \
                3   8
                   / \
                  7   9
        */

        tree.add(5);
        tree.add(3);
        tree.add(7);
        tree.add(8);
        tree.add(9);

        assertEquals(5, tree.size());

        assertEquals((Integer) 5, tree.getRoot().getData());
        assertEquals(2, tree.getRoot().getHeight());
        assertEquals(-1, tree.getRoot().getBalanceFactor());
        assertEquals((Integer) 3, tree.getRoot().getLeft().getData());
        assertEquals(0, tree.getRoot().getLeft().getHeight());
        assertEquals(0, tree.getRoot().getLeft().getBalanceFactor());
        assertEquals((Integer) 8, tree.getRoot().getRight().getData());
        assertEquals(1, tree.getRoot().getRight().getHeight());
        assertEquals(0, tree.getRoot().getRight().getBalanceFactor());
        assertEquals((Integer) 7, tree.getRoot().getRight().getLeft().getData());
        assertEquals(0, tree.getRoot().getRight().getLeft().getHeight());
        assertEquals(0, tree.getRoot().getRight().getLeft().getBalanceFactor());
        assertEquals((Integer) 9, tree.getRoot().getRight().getRight().getData());
        assertEquals(0, tree.getRoot().getRight().getRight().getHeight());
        assertEquals(0, tree.getRoot().getRight().getRight().getBalanceFactor());
    }

    @Test(timeout = TIMEOUT)
    public void testAddLeftRightRotate() {
        /*
                  5
                 / \
                3   7
                     \
                      9
                     /
                    8
        */
        /*
                  5
                 / \
                3   8
                   / \
                  7   9
        */

        tree.add(5);
        tree.add(3);
        tree.add(7);
        tree.add(9);
        tree.add(8);

        assertEquals(5, tree.size());

        assertEquals((Integer) 5, tree.getRoot().getData());
        assertEquals(2, tree.getRoot().getHeight());
        assertEquals(-1, tree.getRoot().getBalanceFactor());
        assertEquals((Integer) 3, tree.getRoot().getLeft().getData());
        assertEquals(0, tree.getRoot().getLeft().getHeight());
        assertEquals(0, tree.getRoot().getLeft().getBalanceFactor());
        assertEquals((Integer) 8, tree.getRoot().getRight().getData());
        assertEquals(1, tree.getRoot().getRight().getHeight());
        assertEquals(0, tree.getRoot().getRight().getBalanceFactor());
        assertEquals((Integer) 7, tree.getRoot().getRight().getLeft().getData());
        assertEquals(0, tree.getRoot().getRight().getLeft().getHeight());
        assertEquals(0, tree.getRoot().getRight().getLeft().getBalanceFactor());
        assertEquals((Integer) 9, tree.getRoot().getRight().getRight().getData());
        assertEquals(0, tree.getRoot().getRight().getRight().getHeight());
        assertEquals(0, tree.getRoot().getRight().getRight().getBalanceFactor());
    }

    @Test(timeout = TIMEOUT)
    public void testAddRightRightRotateAtRoot() {
        /*
                  5
                 /
                3
               /
              2
        */
        /*
                  3
                 / \
                2   5
        */

        tree.add(5);
        tree.add(3);
        tree.add(2);

        assertEquals(3, tree.size());

        assertEquals((Integer) 3, tree.getRoot().getData());
        assertEquals(1, tree.getRoot().getHeight());
        assertEquals(0, tree.getRoot().getBalanceFactor());
        assertEquals((Integer) 2, tree.getRoot().getLeft().getData());
        assertEquals(0, tree.getRoot().getLeft().getHeight());
        assertEquals(0, tree.getRoot().getLeft().getBalanceFactor());
        assertEquals((Integer) 5, tree.getRoot().getRight().getData());
        assertEquals(0, tree.getRoot().getRight().getHeight());
        assertEquals(0, tree.getRoot().getRight().getBalanceFactor());
    }

    @Test(timeout = TIMEOUT)
    public void testAddRightLeftRotateAtRoot() {
        /*
                  5
                 /
                2
                 \
                  3
        */
        /*
                  3
                 / \
                2   5
        */

        tree.add(5);
        tree.add(3);
        tree.add(2);

        assertEquals(3, tree.size());

        assertEquals((Integer) 3, tree.getRoot().getData());
        assertEquals(1, tree.getRoot().getHeight());
        assertEquals(0, tree.getRoot().getBalanceFactor());
        assertEquals((Integer) 2, tree.getRoot().getLeft().getData());
        assertEquals(0, tree.getRoot().getLeft().getHeight());
        assertEquals(0, tree.getRoot().getLeft().getBalanceFactor());
        assertEquals((Integer) 5, tree.getRoot().getRight().getData());
        assertEquals(0, tree.getRoot().getRight().getHeight());
        assertEquals(0, tree.getRoot().getRight().getBalanceFactor());
    }

    @Test(timeout = TIMEOUT)
    public void testAddLeftLeftRotateAtRoot() {
        /*
                  5
                   \
                    7
                     \
                      8
        */
        /*
                  7
                 / \
                5   8
        */

        tree.add(5);
        tree.add(7);
        tree.add(8);

        assertEquals(3, tree.size());

        assertEquals((Integer) 7, tree.getRoot().getData());
        assertEquals(1, tree.getRoot().getHeight());
        assertEquals(0, tree.getRoot().getBalanceFactor());
        assertEquals((Integer) 5, tree.getRoot().getLeft().getData());
        assertEquals(0, tree.getRoot().getLeft().getHeight());
        assertEquals(0, tree.getRoot().getLeft().getBalanceFactor());
        assertEquals((Integer) 8, tree.getRoot().getRight().getData());
        assertEquals(0, tree.getRoot().getRight().getHeight());
        assertEquals(0, tree.getRoot().getRight().getBalanceFactor());
    }

    @Test(timeout = TIMEOUT)
    public void testAddLeftRightRotateAtRoot() {
        /*
                  5
                   \
                    8
                   /
                  7
        */
        /*
                  7
                 / \
                5   8
        */

        tree.add(5);
        tree.add(7);
        tree.add(8);

        assertEquals(3, tree.size());

        assertEquals((Integer) 7, tree.getRoot().getData());
        assertEquals(1, tree.getRoot().getHeight());
        assertEquals(0, tree.getRoot().getBalanceFactor());
        assertEquals((Integer) 5, tree.getRoot().getLeft().getData());
        assertEquals(0, tree.getRoot().getLeft().getHeight());
        assertEquals(0, tree.getRoot().getLeft().getBalanceFactor());
        assertEquals((Integer) 8, tree.getRoot().getRight().getData());
        assertEquals(0, tree.getRoot().getRight().getHeight());
        assertEquals(0, tree.getRoot().getRight().getBalanceFactor());
    }

    @Test(timeout = TIMEOUT, expected = IllegalArgumentException.class)
    public void testAddNull() {
        tree.add(null);
    }

    @Test(timeout = TIMEOUT)
    public void testRemoveLeaf() {
        /*
              5
             / \
            3   7
        */

        tree.add(5);
        tree.add(3);
        tree.add(7);

        assertEquals(3, tree.size());

        tree.remove(3);
        tree.remove(7);

        assertEquals(1, tree.size());
        assertEquals((Integer) 5, tree.getRoot().getData());
        assertEquals(0, tree.getRoot().getHeight());
        assertEquals(0, tree.getRoot().getBalanceFactor());
    }

    @Test(timeout = TIMEOUT)
    public void testRemoveOneChild() {
        /*
              5
             / \
            3   7
           /
          2
        */

                /*
              5
             / \
            2   7
        */

        tree.add(5);
        tree.add(3);
        tree.add(7);
        tree.add(2);

        assertEquals(4, tree.size());

        tree.remove(3);

        assertEquals(3, tree.size());

        assertEquals((Integer) 5, tree.getRoot().getData());
        assertEquals(1, tree.getRoot().getHeight());
        assertEquals(0, tree.getRoot().getBalanceFactor());
        assertEquals((Integer) 2, tree.getRoot().getLeft().getData());
        assertEquals(0, tree.getRoot().getLeft().getHeight());
        assertEquals(0, tree.getRoot().getLeft().getBalanceFactor());
        assertEquals((Integer) 7, tree.getRoot().getRight().getData());
        assertEquals(0, tree.getRoot().getRight().getHeight());
        assertEquals(0, tree.getRoot().getRight().getBalanceFactor());
    }

    @Test(timeout = TIMEOUT)
    public void testRemoveTwoChildren() {
        /*
              5
             / \
            3   7
           / \
          2   4
        */
        /*
              5
             / \
            2   7
             \
              4
        */

        tree.add(5);
        tree.add(3);
        tree.add(7);
        tree.add(2);
        tree.add(4);

        assertEquals(5, tree.size());

        tree.remove(3);

        assertEquals(4, tree.size());

        assertEquals((Integer) 5, tree.getRoot().getData());
        assertEquals(2, tree.getRoot().getHeight());
        assertEquals(1, tree.getRoot().getBalanceFactor());
        assertEquals((Integer) 2, tree.getRoot().getLeft().getData());
        assertEquals(1, tree.getRoot().getLeft().getHeight());
        assertEquals(-1, tree.getRoot().getLeft().getBalanceFactor());
        assertEquals((Integer) 7, tree.getRoot().getRight().getData());
        assertEquals(0, tree.getRoot().getRight().getHeight());
        assertEquals(0, tree.getRoot().getRight().getBalanceFactor());
        assertEquals((Integer) 4, tree.getRoot().getLeft().getRight().getData());
        assertEquals(0, tree.getRoot().getLeft().getRight().getHeight());
        assertEquals(0, tree.getRoot().getLeft().getRight().getBalanceFactor());
    }

    @Test(timeout = TIMEOUT)
    public void testRemoveLeafRightRightRotation() {
        /*
              5
             / \
            3   7
           /
          2
        */

        /*
              5
             /
            3
           /
          2
        */

        /*
              3
             / \
            2   5
        */

        tree.add(5);
        tree.add(3);
        tree.add(7);
        tree.add(2);

        assertEquals(4, tree.size());

        tree.remove(7);

        assertEquals(3, tree.size());

        assertEquals((Integer) 3, tree.getRoot().getData());
        assertEquals(1, tree.getRoot().getHeight());
        assertEquals(0, tree.getRoot().getBalanceFactor());
        assertEquals((Integer) 2, tree.getRoot().getLeft().getData());
        assertEquals(0, tree.getRoot().getLeft().getHeight());
        assertEquals(0, tree.getRoot().getLeft().getBalanceFactor());
        assertEquals((Integer) 5, tree.getRoot().getRight().getData());
        assertEquals(0, tree.getRoot().getRight().getHeight());
        assertEquals(0, tree.getRoot().getRight().getBalanceFactor());
    }

    @Test(timeout = TIMEOUT)
    public void testRemoveLeafRightLeftRotation() {
        /*
              5
             / \
            3   7
             \
              4
        */

        /*
              5
             /
            3
             \
              4
        */

        /*
              4
             / \
            3   5
        */

        tree.add(5);
        tree.add(3);
        tree.add(7);
        tree.add(4);

        assertEquals(4, tree.size());

        tree.remove(7);

        assertEquals(3, tree.size());

        assertEquals((Integer) 4, tree.getRoot().getData());
        assertEquals(1, tree.getRoot().getHeight());
        assertEquals(0, tree.getRoot().getBalanceFactor());
        assertEquals((Integer) 3, tree.getRoot().getLeft().getData());
        assertEquals(0, tree.getRoot().getLeft().getHeight());
        assertEquals(0, tree.getRoot().getLeft().getBalanceFactor());
        assertEquals((Integer) 5, tree.getRoot().getRight().getData());
        assertEquals(0, tree.getRoot().getRight().getHeight());
        assertEquals(0, tree.getRoot().getRight().getBalanceFactor());
    }

    @Test(timeout = TIMEOUT)
    public void testRemoveLeafLeftLeftRotation() {
        /*
              5
             / \
            3   7
                 \
                  8
        */

        /*
              5
               \
                7
                 \
                  8
        */

        /*
              7
             / \
            5   8
        */

        tree.add(5);
        tree.add(3);
        tree.add(7);
        tree.add(8);

        assertEquals(4, tree.size());

        tree.remove(3);

        assertEquals(3, tree.size());

        assertEquals((Integer) 7, tree.getRoot().getData());
        assertEquals(1, tree.getRoot().getHeight());
        assertEquals(0, tree.getRoot().getBalanceFactor());
        assertEquals((Integer) 5, tree.getRoot().getLeft().getData());
        assertEquals(0, tree.getRoot().getLeft().getHeight());
        assertEquals(0, tree.getRoot().getLeft().getBalanceFactor());
        assertEquals((Integer) 8, tree.getRoot().getRight().getData());
        assertEquals(0, tree.getRoot().getRight().getHeight());
        assertEquals(0, tree.getRoot().getRight().getBalanceFactor());
    }

    @Test(timeout = TIMEOUT)
    public void testRemoveLeafLeftRightRotation() {
        /*
              5
             / \
            3   7
               /
              6
        */

        /*
              5
               \
                7
               /
              6
        */

        /*
              6
             / \
            5   7
        */

        tree.add(5);
        tree.add(3);
        tree.add(7);
        tree.add(6);

        assertEquals(4, tree.size());

        tree.remove(3);

        assertEquals(3, tree.size());

        assertEquals((Integer) 6, tree.getRoot().getData());
        assertEquals(1, tree.getRoot().getHeight());
        assertEquals(0, tree.getRoot().getBalanceFactor());
        assertEquals((Integer) 5, tree.getRoot().getLeft().getData());
        assertEquals(0, tree.getRoot().getLeft().getHeight());
        assertEquals(0, tree.getRoot().getLeft().getBalanceFactor());
        assertEquals((Integer) 7, tree.getRoot().getRight().getData());
        assertEquals(0, tree.getRoot().getRight().getHeight());
        assertEquals(0, tree.getRoot().getRight().getBalanceFactor());
    }

    @Test(timeout = TIMEOUT)
    public void testRemoveOneChildRightRightRotation() {
        /*
                  5
                 / \
                2   7
               / \   \
              1   3   8
             /
            0
        */

        /*
                  5
                 / \
                2   8
               / \
              1   3
             /
            0
        */

        /*
                  2
                 / \
                1   5
               /   / \
              0   3   8
        */

        tree.add(5);
        tree.add(2);
        tree.add(7);
        tree.add(1);
        tree.add(3);
        tree.add(8);
        tree.add(0);

        assertEquals(7, tree.size());

        tree.remove(7);

        assertEquals(6, tree.size());

        assertEquals((Integer) 2, tree.getRoot().getData());
        assertEquals(2, tree.getRoot().getHeight());
        assertEquals(0, tree.getRoot().getBalanceFactor());
        assertEquals((Integer) 1, tree.getRoot().getLeft().getData());
        assertEquals(1, tree.getRoot().getLeft().getHeight());
        assertEquals(1, tree.getRoot().getLeft().getBalanceFactor());
        assertEquals((Integer) 5, tree.getRoot().getRight().getData());
        assertEquals(1, tree.getRoot().getRight().getHeight());
        assertEquals(0, tree.getRoot().getRight().getBalanceFactor());
        assertEquals((Integer) 0, tree.getRoot().getLeft().getLeft().getData());
        assertEquals(0, tree.getRoot().getLeft().getLeft().getHeight());
        assertEquals(0, tree.getRoot().getLeft().getLeft().getBalanceFactor());
        assertEquals((Integer) 3, tree.getRoot().getRight().getLeft().getData());
        assertEquals(0, tree.getRoot().getRight().getLeft().getHeight());
        assertEquals(0, tree.getRoot().getRight().getLeft().getBalanceFactor());
        assertEquals((Integer) 8, tree.getRoot().getRight().getRight().getData());
        assertEquals(0, tree.getRoot().getRight().getRight().getHeight());
        assertEquals(0, tree.getRoot().getRight().getRight().getBalanceFactor());
    }

    @Test(timeout = TIMEOUT)
    public void testRemoveOneChildRightLeftRotation() {
        /*
                  5
                 / \
                2   7
               / \   \
              1   3   8
                   \
                    4
        */

        /*
                  5
                 / \
                2   8
               / \
              1   3
                   \
                    4
        */

        /*
                  3
                 / \
                2   5
               /   / \
              1   4   8
        */

        tree.add(5);
        tree.add(2);
        tree.add(7);
        tree.add(1);
        tree.add(3);
        tree.add(8);
        tree.add(4);

        assertEquals(7, tree.size());

        tree.remove(7);

        assertEquals(6, tree.size());

        assertEquals((Integer) 3, tree.getRoot().getData());
        assertEquals(2, tree.getRoot().getHeight());
        assertEquals(0, tree.getRoot().getBalanceFactor());
        assertEquals((Integer) 2, tree.getRoot().getLeft().getData());
        assertEquals(1, tree.getRoot().getLeft().getHeight());
        assertEquals(1, tree.getRoot().getLeft().getBalanceFactor());
        assertEquals((Integer) 5, tree.getRoot().getRight().getData());
        assertEquals(1, tree.getRoot().getRight().getHeight());
        assertEquals(0, tree.getRoot().getRight().getBalanceFactor());
        assertEquals((Integer) 1, tree.getRoot().getLeft().getLeft().getData());
        assertEquals(0, tree.getRoot().getLeft().getLeft().getHeight());
        assertEquals(0, tree.getRoot().getLeft().getLeft().getBalanceFactor());
        assertEquals((Integer) 4, tree.getRoot().getRight().getLeft().getData());
        assertEquals(0, tree.getRoot().getRight().getLeft().getHeight());
        assertEquals(0, tree.getRoot().getRight().getLeft().getBalanceFactor());
        assertEquals((Integer) 8, tree.getRoot().getRight().getRight().getData());
        assertEquals(0, tree.getRoot().getRight().getRight().getHeight());
        assertEquals(0, tree.getRoot().getRight().getRight().getBalanceFactor());
    }

    @Test(timeout = TIMEOUT)
    public void testRemoveOneChildLeftLeftRotation() {
        /*
                  5
                 / \
                2   8
               /   / \
              1   7   9
                       \
                       10
        */

        /*
                  5
                 / \
                1   8
                   / \
                  7   9
                       \
                       10
        */

        /*
                  8
                 / \
                5   9
               / \   \
              1   7   10
        */

        tree.add(5);
        tree.add(2);
        tree.add(8);
        tree.add(1);
        tree.add(7);
        tree.add(9);
        tree.add(10);

        assertEquals(7, tree.size());

        tree.remove(2   );

        assertEquals(6, tree.size());

        assertEquals((Integer) 8, tree.getRoot().getData());
        assertEquals(2, tree.getRoot().getHeight());
        assertEquals(0, tree.getRoot().getBalanceFactor());
        assertEquals((Integer) 5, tree.getRoot().getLeft().getData());
        assertEquals(1, tree.getRoot().getLeft().getHeight());
        assertEquals(0, tree.getRoot().getLeft().getBalanceFactor());
        assertEquals((Integer) 9, tree.getRoot().getRight().getData());
        assertEquals(1, tree.getRoot().getRight().getHeight());
        assertEquals(-1, tree.getRoot().getRight().getBalanceFactor());
        assertEquals((Integer) 1, tree.getRoot().getLeft().getLeft().getData());
        assertEquals(0, tree.getRoot().getLeft().getLeft().getHeight());
        assertEquals(0, tree.getRoot().getLeft().getLeft().getBalanceFactor());
        assertEquals((Integer) 7, tree.getRoot().getLeft().getRight().getData());
        assertEquals(0, tree.getRoot().getLeft().getRight().getHeight());
        assertEquals(0, tree.getRoot().getLeft().getRight().getBalanceFactor());
        assertEquals((Integer) 10, tree.getRoot().getRight().getRight().getData());
        assertEquals(0, tree.getRoot().getRight().getRight().getHeight());
        assertEquals(0, tree.getRoot().getRight().getRight().getBalanceFactor());
    }

    @Test(timeout = TIMEOUT)
    public void testRemoveOneChildLeftRightRotation() {
        /*
                  5
                 / \
                2   8
               /   / \
              1   7   9
                 /
                6
        */

        /*
                  5
                 / \
                1   8
                   / \
                  7   9
                 /
                6
        */

        /*
                  7
                 / \
                5   8
               / \   \
              1   6   9
        */

        tree.add(5);
        tree.add(2);
        tree.add(8);
        tree.add(1);
        tree.add(7);
        tree.add(9);
        tree.add(6);

        assertEquals(7, tree.size());

        tree.remove(2   );

        assertEquals(6, tree.size());

        assertEquals((Integer) 7, tree.getRoot().getData());
        assertEquals(2, tree.getRoot().getHeight());
        assertEquals(0, tree.getRoot().getBalanceFactor());
        assertEquals((Integer) 5, tree.getRoot().getLeft().getData());
        assertEquals(1, tree.getRoot().getLeft().getHeight());
        assertEquals(0, tree.getRoot().getLeft().getBalanceFactor());
        assertEquals((Integer) 8, tree.getRoot().getRight().getData());
        assertEquals(1, tree.getRoot().getRight().getHeight());
        assertEquals(-1, tree.getRoot().getRight().getBalanceFactor());
        assertEquals((Integer) 1, tree.getRoot().getLeft().getLeft().getData());
        assertEquals(0, tree.getRoot().getLeft().getLeft().getHeight());
        assertEquals(0, tree.getRoot().getLeft().getLeft().getBalanceFactor());
        assertEquals((Integer) 6, tree.getRoot().getLeft().getRight().getData());
        assertEquals(0, tree.getRoot().getLeft().getRight().getHeight());
        assertEquals(0, tree.getRoot().getLeft().getRight().getBalanceFactor());
        assertEquals((Integer) 9, tree.getRoot().getRight().getRight().getData());
        assertEquals(0, tree.getRoot().getRight().getRight().getHeight());
        assertEquals(0, tree.getRoot().getRight().getRight().getBalanceFactor());
    }

    @Test(timeout = TIMEOUT)
    public void testRemoveTwoChildrenRightRightRotation() {
        tree.add(8);
        tree.add(5);
        tree.add(11);
        tree.add(3);
        tree.add(6);
        tree.add(10);
        tree.add(12);
        tree.add(2);
        tree.add(4);
        tree.add(7);
        tree.add(9);
        tree.add(1);

        assertEquals(12, tree.size());

        tree.remove(8);

        assertEquals(11, tree.size());

        assertEquals((Integer) 7, tree.getRoot().getData());
        assertEquals(3, tree.getRoot().getHeight());
        assertEquals(0, tree.getRoot().getBalanceFactor());
        assertEquals((Integer) 3, tree.getRoot().getLeft().getData());
        assertEquals(2, tree.getRoot().getLeft().getHeight());
        assertEquals(0, tree.getRoot().getLeft().getBalanceFactor());
        assertEquals((Integer) 11, tree.getRoot().getRight().getData());
        assertEquals(2, tree.getRoot().getRight().getHeight());
        assertEquals(1, tree.getRoot().getRight().getBalanceFactor());
        assertEquals((Integer) 2, tree.getRoot().getLeft().getLeft().getData());
        assertEquals(1, tree.getRoot().getLeft().getLeft().getHeight());
        assertEquals(1, tree.getRoot().getLeft().getLeft().getBalanceFactor());
        assertEquals((Integer) 5, tree.getRoot().getLeft().getRight().getData());
        assertEquals(1, tree.getRoot().getLeft().getRight().getHeight());
        assertEquals(0, tree.getRoot().getLeft().getRight().getBalanceFactor());
        assertEquals((Integer) 10, tree.getRoot().getRight().getLeft().getData());
        assertEquals(1, tree.getRoot().getRight().getLeft().getHeight());
        assertEquals(1, tree.getRoot().getRight().getLeft().getBalanceFactor());
        assertEquals((Integer) 12, tree.getRoot().getRight().getRight().getData());
        assertEquals(0, tree.getRoot().getRight().getRight().getHeight());
        assertEquals(0, tree.getRoot().getRight().getRight().getBalanceFactor());
        assertEquals((Integer) 1, tree.getRoot().getLeft().getLeft().getLeft().getData());
        assertEquals(0, tree.getRoot().getLeft().getLeft().getLeft().getHeight());
        assertEquals(0, tree.getRoot().getLeft().getLeft().getLeft().getBalanceFactor());
        assertEquals((Integer) 4, tree.getRoot().getLeft().getRight().getLeft().getData());
        assertEquals(0, tree.getRoot().getLeft().getRight().getLeft().getHeight());
        assertEquals(0, tree.getRoot().getLeft().getRight().getLeft().getBalanceFactor());
        assertEquals((Integer) 6, tree.getRoot().getLeft().getRight().getRight().getData());
        assertEquals(0, tree.getRoot().getLeft().getRight().getRight().getHeight());
        assertEquals(0, tree.getRoot().getLeft().getRight().getRight().getBalanceFactor());
        assertEquals((Integer) 9, tree.getRoot().getRight().getLeft().getLeft().getData());
        assertEquals(0, tree.getRoot().getRight().getLeft().getLeft().getHeight());
        assertEquals(0, tree.getRoot().getRight().getLeft().getLeft().getBalanceFactor());
    }

    @Test(timeout = TIMEOUT)
    public void testRemoveTwoChildrenRightLeftRotation() {
        tree.add(8);
        tree.add(3);
        tree.add(11);
        tree.add(2);
        tree.add(5);
        tree.add(10);
        tree.add(12);
        tree.add(1);
        tree.add(4);
        tree.add(6);
        tree.add(9);
        tree.add(7);

        assertEquals(12, tree.size());

        tree.remove(8);

        assertEquals(11, tree.size());

        assertEquals((Integer) 7, tree.getRoot().getData());
        assertEquals(3, tree.getRoot().getHeight());
        assertEquals(0, tree.getRoot().getBalanceFactor());
        assertEquals((Integer) 3, tree.getRoot().getLeft().getData());
        assertEquals(2, tree.getRoot().getLeft().getHeight());
        assertEquals(0, tree.getRoot().getLeft().getBalanceFactor());
        assertEquals((Integer) 11, tree.getRoot().getRight().getData());
        assertEquals(2, tree.getRoot().getRight().getHeight());
        assertEquals(1, tree.getRoot().getRight().getBalanceFactor());
        assertEquals((Integer) 2, tree.getRoot().getLeft().getLeft().getData());
        assertEquals(1, tree.getRoot().getLeft().getLeft().getHeight());
        assertEquals(1, tree.getRoot().getLeft().getLeft().getBalanceFactor());
        assertEquals((Integer) 5, tree.getRoot().getLeft().getRight().getData());
        assertEquals(1, tree.getRoot().getLeft().getRight().getHeight());
        assertEquals(0, tree.getRoot().getLeft().getRight().getBalanceFactor());
        assertEquals((Integer) 10, tree.getRoot().getRight().getLeft().getData());
        assertEquals(1, tree.getRoot().getRight().getLeft().getHeight());
        assertEquals(1, tree.getRoot().getRight().getLeft().getBalanceFactor());
        assertEquals((Integer) 12, tree.getRoot().getRight().getRight().getData());
        assertEquals(0, tree.getRoot().getRight().getRight().getHeight());
        assertEquals(0, tree.getRoot().getRight().getRight().getBalanceFactor());
        assertEquals((Integer) 1, tree.getRoot().getLeft().getLeft().getLeft().getData());
        assertEquals(0, tree.getRoot().getLeft().getLeft().getLeft().getHeight());
        assertEquals(0, tree.getRoot().getLeft().getLeft().getLeft().getBalanceFactor());
        assertEquals((Integer) 4, tree.getRoot().getLeft().getRight().getLeft().getData());
        assertEquals(0, tree.getRoot().getLeft().getRight().getLeft().getHeight());
        assertEquals(0, tree.getRoot().getLeft().getRight().getLeft().getBalanceFactor());
        assertEquals((Integer) 6, tree.getRoot().getLeft().getRight().getRight().getData());
        assertEquals(0, tree.getRoot().getLeft().getRight().getRight().getHeight());
        assertEquals(0, tree.getRoot().getLeft().getRight().getRight().getBalanceFactor());
        assertEquals((Integer) 9, tree.getRoot().getRight().getLeft().getLeft().getData());
        assertEquals(0, tree.getRoot().getRight().getLeft().getLeft().getHeight());
        assertEquals(0, tree.getRoot().getRight().getLeft().getLeft().getBalanceFactor());
    }

    @Test(timeout = TIMEOUT)
    public void testRemoveTwoChildrenLeftLeftRotation() {
        tree.add(5);
        tree.add(2);
        tree.add(8);
        tree.add(1);
        tree.add(3);
        tree.add(7);
        tree.add(10);
        tree.add(4);
        tree.add(6);
        tree.add(9);
        tree.add(11);
        tree.add(12);

        assertEquals(12, tree.size());

        tree.remove(5);

        assertEquals(11, tree.size());

        assertEquals((Integer) 8, tree.getRoot().getData());
        assertEquals(3, tree.getRoot().getHeight());
        assertEquals(0, tree.getRoot().getBalanceFactor());
        assertEquals((Integer) 4, tree.getRoot().getLeft().getData());
        assertEquals(2, tree.getRoot().getLeft().getHeight());
        assertEquals(0, tree.getRoot().getLeft().getBalanceFactor());
        assertEquals((Integer) 10, tree.getRoot().getRight().getData());
        assertEquals(2, tree.getRoot().getRight().getHeight());
        assertEquals(-1, tree.getRoot().getRight().getBalanceFactor());
        assertEquals((Integer) 2, tree.getRoot().getLeft().getLeft().getData());
        assertEquals(1, tree.getRoot().getLeft().getLeft().getHeight());
        assertEquals(0, tree.getRoot().getLeft().getLeft().getBalanceFactor());
        assertEquals((Integer) 7, tree.getRoot().getLeft().getRight().getData());
        assertEquals(1, tree.getRoot().getLeft().getRight().getHeight());
        assertEquals(1, tree.getRoot().getLeft().getRight().getBalanceFactor());
        assertEquals((Integer) 9, tree.getRoot().getRight().getLeft().getData());
        assertEquals(0, tree.getRoot().getRight().getLeft().getHeight());
        assertEquals(0, tree.getRoot().getRight().getLeft().getBalanceFactor());
        assertEquals((Integer) 11, tree.getRoot().getRight().getRight().getData());
        assertEquals(1, tree.getRoot().getRight().getRight().getHeight());
        assertEquals(-1, tree.getRoot().getRight().getRight().getBalanceFactor());
        assertEquals((Integer) 1, tree.getRoot().getLeft().getLeft().getLeft().getData());
        assertEquals(0, tree.getRoot().getLeft().getLeft().getLeft().getHeight());
        assertEquals(0, tree.getRoot().getLeft().getLeft().getLeft().getBalanceFactor());
        assertEquals((Integer) 3, tree.getRoot().getLeft().getLeft().getRight().getData());
        assertEquals(0, tree.getRoot().getLeft().getLeft().getRight().getHeight());
        assertEquals(0, tree.getRoot().getLeft().getLeft().getRight().getBalanceFactor());
        assertEquals((Integer) 6, tree.getRoot().getLeft().getRight().getLeft().getData());
        assertEquals(0, tree.getRoot().getLeft().getRight().getLeft().getHeight());
        assertEquals(0, tree.getRoot().getLeft().getRight().getLeft().getBalanceFactor());
        assertEquals((Integer) 12, tree.getRoot().getRight().getRight().getRight().getData());
        assertEquals(0, tree.getRoot().getRight().getRight().getRight().getHeight());
        assertEquals(0, tree.getRoot().getRight().getRight().getRight().getBalanceFactor());
    }

    @Test(timeout = TIMEOUT)
    public void testRemoveTwoChildrenLeftRightRotation() {
        tree.add(5);
        tree.add(2);
        tree.add(10);
        tree.add(1);
        tree.add(3);
        tree.add(8);
        tree.add(11);
        tree.add(4);
        tree.add(7);
        tree.add(9);
        tree.add(12);
        tree.add(6);

        assertEquals(12, tree.size());

        tree.remove(5);

        assertEquals(11, tree.size());

        assertEquals((Integer) 8, tree.getRoot().getData());
        assertEquals(3, tree.getRoot().getHeight());
        assertEquals(0, tree.getRoot().getBalanceFactor());
        assertEquals((Integer) 4, tree.getRoot().getLeft().getData());
        assertEquals(2, tree.getRoot().getLeft().getHeight());
        assertEquals(0, tree.getRoot().getLeft().getBalanceFactor());
        assertEquals((Integer) 10, tree.getRoot().getRight().getData());
        assertEquals(2, tree.getRoot().getRight().getHeight());
        assertEquals(-1, tree.getRoot().getRight().getBalanceFactor());
        assertEquals((Integer) 2, tree.getRoot().getLeft().getLeft().getData());
        assertEquals(1, tree.getRoot().getLeft().getLeft().getHeight());
        assertEquals(0, tree.getRoot().getLeft().getLeft().getBalanceFactor());
        assertEquals((Integer) 7, tree.getRoot().getLeft().getRight().getData());
        assertEquals(1, tree.getRoot().getLeft().getRight().getHeight());
        assertEquals(1, tree.getRoot().getLeft().getRight().getBalanceFactor());
        assertEquals((Integer) 9, tree.getRoot().getRight().getLeft().getData());
        assertEquals(0, tree.getRoot().getRight().getLeft().getHeight());
        assertEquals(0, tree.getRoot().getRight().getLeft().getBalanceFactor());
        assertEquals((Integer) 11, tree.getRoot().getRight().getRight().getData());
        assertEquals(1, tree.getRoot().getRight().getRight().getHeight());
        assertEquals(-1, tree.getRoot().getRight().getRight().getBalanceFactor());
        assertEquals((Integer) 1, tree.getRoot().getLeft().getLeft().getLeft().getData());
        assertEquals(0, tree.getRoot().getLeft().getLeft().getLeft().getHeight());
        assertEquals(0, tree.getRoot().getLeft().getLeft().getLeft().getBalanceFactor());
        assertEquals((Integer) 3, tree.getRoot().getLeft().getLeft().getRight().getData());
        assertEquals(0, tree.getRoot().getLeft().getLeft().getRight().getHeight());
        assertEquals(0, tree.getRoot().getLeft().getLeft().getRight().getBalanceFactor());
        assertEquals((Integer) 6, tree.getRoot().getLeft().getRight().getLeft().getData());
        assertEquals(0, tree.getRoot().getLeft().getRight().getLeft().getHeight());
        assertEquals(0, tree.getRoot().getLeft().getRight().getLeft().getBalanceFactor());
        assertEquals((Integer) 12, tree.getRoot().getRight().getRight().getRight().getData());
        assertEquals(0, tree.getRoot().getRight().getRight().getRight().getHeight());
        assertEquals(0, tree.getRoot().getRight().getRight().getRight().getBalanceFactor());
    }

    @Test(timeout = TIMEOUT, expected = IllegalArgumentException.class)
    public void testRemoveNull() {
        tree.remove(null);
    }

    @Test(timeout = TIMEOUT, expected = NoSuchElementException.class)
    public void testRemoveEmpty() {
        tree.remove(0);
    }

    @Test(timeout = TIMEOUT, expected = NoSuchElementException.class)
    public void testRemoveNotInTree() {
        tree.add(5);
        tree.add(3);
        tree.add(7);

        assertEquals(3, tree.size());

        tree.remove(2);
    }

    @Test(timeout = TIMEOUT)
    public void testGet() {
        tree.add(5);
        tree.add(3);
        tree.add(7);

        assertEquals(3, tree.size());

        assertEquals((Integer) 5, tree.get(5));
        assertEquals((Integer) 3, tree.get(3));
        assertEquals((Integer) 7, tree.get(7));
    }

    @Test(timeout = TIMEOUT, expected = NoSuchElementException.class)
    public void testGetNotInTree() {
        tree.add(5);
        tree.add(3);
        tree.add(7);

        assertEquals(3, tree.size());

        tree.get(2);
    }

    @Test(timeout = TIMEOUT, expected = IllegalArgumentException.class)
    public void testGetNull() {
        tree.add(5);
        tree.add(3);
        tree.add(7);

        assertEquals(3, tree.size());

        tree.get(null);
    }

    @Test(timeout = TIMEOUT)
    public void testContains() {
        tree.add(5);
        tree.add(3);
        tree.add(7);

        assertEquals(3, tree.size());

        assertTrue(tree.contains(5));
        assertTrue(tree.contains(3));
        assertTrue(tree.contains(7));
        assertTrue(!tree.contains(2));
    }

    @Test(timeout = TIMEOUT, expected = IllegalArgumentException.class)
    public void testContainsNull() {
        tree.add(5);
        tree.add(3);
        tree.add(7);

        assertEquals(3, tree.size());

        tree.contains(null);
    }

    @Test(timeout = TIMEOUT)
    public void testHeight() {
        assertEquals(-1, tree.height());

        tree.add(5);
        tree.add(3);
        tree.add(7);

        assertEquals(1, tree.height());
    }

    @Test(timeout = TIMEOUT)
    public void testClear() {
        tree.add(5);
        tree.add(3);
        tree.add(7);

        assertEquals(3, tree.size());

        tree.clear();

        assertEquals(0, tree.size());
        assertEquals(null, tree.getRoot());
    }

    @Test(timeout = TIMEOUT)
    public void testDeepestBranches() {
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

        tree.add(10);
        tree.add(5);
        tree.add(15);
        tree.add(2);
        tree.add(7);
        tree.add(13);
        tree.add(20);
        tree.add(1);
        tree.add(4);
        tree.add(6);
        tree.add(8);
        tree.add(14);
        tree.add(17);
        tree.add(25);
        tree.add(0);
        tree.add(9);
        tree.add(30);

        List<Integer> expected = new ArrayList<>();
        expected.add(10);
        expected.add(5);
        expected.add(2);
        expected.add(1);
        expected.add(0);
        expected.add(7);
        expected.add(8);
        expected.add(9);
        expected.add(15);
        expected.add(20);
        expected.add(25);
        expected.add(30);

        assertEquals(expected, tree.deepestBranches());
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

        tree.add(10);
        tree.add(5);
        tree.add(15);
        tree.add(2);
        tree.add(7);
        tree.add(13);
        tree.add(20);
        tree.add(1);
        tree.add(4);
        tree.add(6);
        tree.add(8);
        tree.add(14);
        tree.add(17);
        tree.add(25);
        tree.add(0);
        tree.add(9);
        tree.add(30);

        List<Integer> expected = new ArrayList<>();
        expected.add(8);
        expected.add(9);
        expected.add(10);
        expected.add(13);

        assertEquals(expected, tree.sortedInBetween(
                7, 14));
    }

    @Test(timeout = TIMEOUT, expected = IllegalArgumentException.class)
    public void testSortedInBetweenNullData1() {
        tree.add(5);
        tree.add(3);
        tree.add(7);

        assertEquals(3, tree.size());

        tree.sortedInBetween(null, 5);
    }

    @Test(timeout = TIMEOUT, expected = IllegalArgumentException.class)
    public void testSortedInBetweenNullData2() {
        tree.add(5);
        tree.add(3);
        tree.add(7);

        assertEquals(3, tree.size());

        tree.sortedInBetween(5, null);
    }

    @Test(timeout = TIMEOUT, expected = IllegalArgumentException.class)
    public void testSortedInBetweenBothNullData() {
        tree.add(5);
        tree.add(3);
        tree.add(7);

        assertEquals(3, tree.size());

        tree.sortedInBetween(null, null);
    }
}
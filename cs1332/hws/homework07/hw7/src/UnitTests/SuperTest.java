import org.junit.Before;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.NoSuchElementException;

import static org.junit.Assert.*;

/**
 * @author Rishi Soni
 * @version 1.0
 */
public class SuperTest {
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
    public void constructorException() {
        Collection<Integer> items = new ArrayList<>();
        items.add(null);

        assertThrows(IllegalArgumentException.class, () -> {
            tree = new AVL<>(items);
        });

        assertThrows(IllegalArgumentException.class, () -> {
            tree = new AVL<>(null);
        });
    }

    @Test(timeout = TIMEOUT)
    public void addException() {
        assertThrows(IllegalArgumentException.class, () -> {
            tree.add(null);
        });
    }

    @Test(timeout = TIMEOUT)
    public void removeException() {
        assertThrows(IllegalArgumentException.class, () -> {
            tree.remove(null);
        });

        tree.add(1);
        tree.add(2);
        tree.add(3);

        assertThrows(NoSuchElementException.class, () -> {
            tree.remove(4);
        });
    }

    @Test(timeout = TIMEOUT)
    public void getException() {
        assertThrows(IllegalArgumentException.class, () -> {
            tree.get(null);
        });

        tree.add(1);
        tree.add(2);
        tree.add(3);
        tree.add(-3);
        tree.add(-7);

        assertThrows(NoSuchElementException.class, () -> {
            tree.get(-5);
        });
    }

    @Test(timeout = TIMEOUT)
    public void containsException() {
        assertThrows(IllegalArgumentException.class, () -> {
            tree.contains(null);
        });
    }

    @Test(timeout = TIMEOUT)
    public void sortedInBetweenException() {
        assertThrows(IllegalArgumentException.class, () -> {
            tree.sortedInBetween(null, 1);
        });

        assertThrows(IllegalArgumentException.class, () -> {
            tree.sortedInBetween(1, null);
        });

        assertThrows(IllegalArgumentException.class, () -> {
            tree.sortedInBetween(5, 1);
        });
    }

    @Test(timeout = TIMEOUT)
    public void initializationWithCollection() {
        Collection<Integer> items = new ArrayList<>();
        items.add(2);
        items.add(1);
        items.add(3);

        tree = new AVL<>(items);

        assertEquals(1, tree.height());
        assertEquals(3, tree.size());
        assertEquals(1, tree.getRoot().getHeight());
        assertEquals(0, tree.getRoot().getBalanceFactor());
        assertEquals((Integer) 2, tree.getRoot().getData());

        AVLNode<Integer> left = tree.getRoot().getLeft();
        assertEquals((Integer) 1, left.getData());
        assertEquals(0, left.getHeight());
        assertEquals(0, left.getBalanceFactor());

        AVLNode<Integer> right = tree.getRoot().getRight();
        assertEquals((Integer) 3, right.getData());
        assertEquals(0, right.getHeight());
        assertEquals(0, right.getBalanceFactor());
    }

    @Test(timeout = TIMEOUT)
    public void addLeftRotate() {
        tree.add(0);
        tree.add(10);
        tree.add(20);

        assertEquals(1, tree.height());
        assertEquals(3, tree.size());
        assertEquals(1, tree.getRoot().getHeight());
        assertEquals(0, tree.getRoot().getBalanceFactor());
        assertEquals((Integer) 10, tree.getRoot().getData());

        AVLNode<Integer> left = tree.getRoot().getLeft();
        assertEquals((Integer) 0, left.getData());
        assertEquals(0, left.getHeight());
        assertEquals(0, left.getBalanceFactor());

        AVLNode<Integer> right = tree.getRoot().getRight();
        assertEquals((Integer) 20, right.getData());
        assertEquals(0, right.getHeight());
        assertEquals(0, right.getBalanceFactor());
    }

    @Test(timeout = TIMEOUT)
    public void addRightRotate() {
        tree.add(10);
        tree.add(5);
        tree.add(1);

        assertEquals(1, tree.height());
        assertEquals(3, tree.size());
        assertEquals(1, tree.getRoot().getHeight());
        assertEquals(0, tree.getRoot().getBalanceFactor());
        assertEquals((Integer) 5, tree.getRoot().getData());

        AVLNode<Integer> left = tree.getRoot().getLeft();
        assertEquals((Integer) 1, left.getData());
        assertEquals(0, left.getHeight());
        assertEquals(0, left.getBalanceFactor());

        AVLNode<Integer> right = tree.getRoot().getRight();
        assertEquals((Integer) 10, right.getData());
        assertEquals(0, right.getHeight());
        assertEquals(0, right.getBalanceFactor());
    }

    @Test(timeout = TIMEOUT)
    public void addLeftRightRotate() {
        tree.add(10);
        tree.add(5);
        tree.add(7);

        assertEquals(1, tree.height());
        assertEquals(3, tree.size());
        assertEquals(1, tree.getRoot().getHeight());
        assertEquals(0, tree.getRoot().getBalanceFactor());
        assertEquals((Integer) 7, tree.getRoot().getData());

        AVLNode<Integer> left = tree.getRoot().getLeft();
        assertEquals((Integer) 5, left.getData());
        assertEquals(0, left.getHeight());
        assertEquals(0, left.getBalanceFactor());

        AVLNode<Integer> right = tree.getRoot().getRight();
        assertEquals((Integer) 10, right.getData());
        assertEquals(0, right.getHeight());
        assertEquals(0, right.getBalanceFactor());
    }

    @Test(timeout = TIMEOUT)
    public void addRightLeftRotate() {
        tree.add(0);
        tree.add(10);
        tree.add(7);

        assertEquals(1, tree.height());
        assertEquals(3, tree.size());
        assertEquals(1, tree.getRoot().getHeight());
        assertEquals(0, tree.getRoot().getBalanceFactor());
        assertEquals((Integer) 7, tree.getRoot().getData());

        AVLNode<Integer> left = tree.getRoot().getLeft();
        assertEquals((Integer) 0, left.getData());
        assertEquals(0, left.getHeight());
        assertEquals(0, left.getBalanceFactor());

        AVLNode<Integer> right = tree.getRoot().getRight();
        assertEquals((Integer) 10, right.getData());
        assertEquals(0, right.getHeight());
        assertEquals(0, right.getBalanceFactor());
    }

    @Test(timeout = TIMEOUT)
    public void addComplicated() {
        //Right rotation at root
        tree.add(40);
        tree.add(50);
        tree.add(20);
        tree.add(10);
        tree.add(25);
        tree.add(5);

        assertEquals(2, tree.height());
        assertEquals(6, tree.size());
        assertEquals(2, tree.getRoot().getHeight());
        assertEquals(0, tree.getRoot().getBalanceFactor());
        assertEquals((Integer) 20, tree.getRoot().getData());

        AVLNode<Integer> left = tree.getRoot().getLeft();
        assertEquals((Integer) 10, left.getData());
        assertEquals(1, left.getHeight());
        assertEquals(1, left.getBalanceFactor());

        left = left.getLeft();
        assertEquals((Integer) 5, left.getData());
        assertEquals(0, left.getHeight());
        assertEquals(0, left.getBalanceFactor());

        AVLNode<Integer> right = tree.getRoot().getRight();
        assertEquals((Integer) 40, right.getData());
        assertEquals(1, right.getHeight());
        assertEquals(0, right.getBalanceFactor());

        left = right.getLeft();
        assertEquals((Integer) 25, left.getData());
        assertEquals(0, left.getHeight());
        assertEquals(0, left.getBalanceFactor());

        right = right.getRight();
        assertEquals((Integer) 50, right.getData());
        assertEquals(0, right.getHeight());
        assertEquals(0, right.getBalanceFactor());

        //Nothing should change since these are duplicates
        tree.add(20);
        tree.add(25);
        tree.add(5);

        assertEquals(2, tree.height());
        assertEquals(6, tree.size()); //Size should not change
        assertEquals(2, tree.getRoot().getHeight());
        assertEquals(0, tree.getRoot().getBalanceFactor());
        assertEquals((Integer) 20, tree.getRoot().getData());

        left = tree.getRoot().getLeft();
        assertEquals((Integer) 10, left.getData());
        assertEquals(1, left.getHeight());
        assertEquals(1, left.getBalanceFactor());

        left = left.getLeft();
        assertEquals((Integer) 5, left.getData());
        assertEquals(0, left.getHeight());
        assertEquals(0, left.getBalanceFactor());

        right = tree.getRoot().getRight();
        assertEquals((Integer) 40, right.getData());
        assertEquals(1, right.getHeight());
        assertEquals(0, right.getBalanceFactor());

        left = right.getLeft();
        assertEquals((Integer) 25, left.getData());
        assertEquals(0, left.getHeight());
        assertEquals(0, left.getBalanceFactor());

        right = right.getRight();
        assertEquals((Integer) 50, right.getData());
        assertEquals(0, right.getHeight());
        assertEquals(0, right.getBalanceFactor());

        //Left Right Rotate
        tree.add(7);

        assertEquals(2, tree.height());
        assertEquals(7, tree.size());
        assertEquals(2, tree.getRoot().getHeight());
        assertEquals(0, tree.getRoot().getBalanceFactor());
        assertEquals((Integer) 20, tree.getRoot().getData());

        left = tree.getRoot().getLeft();
        assertEquals((Integer) 7, left.getData());
        assertEquals(1, left.getHeight());
        assertEquals(0, left.getBalanceFactor());

        left = left.getLeft();
        assertEquals((Integer) 5, left.getData());
        assertEquals(0, left.getHeight());
        assertEquals(0, left.getBalanceFactor());

        right = tree.getRoot().getLeft().getRight();
        assertEquals((Integer) 10, right.getData());
        assertEquals(0, right.getHeight());
        assertEquals(0, right.getBalanceFactor());

        right = tree.getRoot().getRight();
        assertEquals((Integer) 40, right.getData());
        assertEquals(1, right.getHeight());
        assertEquals(0, right.getBalanceFactor());

        left = right.getLeft();
        assertEquals((Integer) 25, left.getData());
        assertEquals(0, left.getHeight());
        assertEquals(0, left.getBalanceFactor());

        right = right.getRight();
        assertEquals((Integer) 50, right.getData());
        assertEquals(0, right.getHeight());
        assertEquals(0, right.getBalanceFactor());
    }

    @Test(timeout = TIMEOUT)
    public void removeLeftRotate() {
        Integer a = 10;

        tree.add(a);
        tree.add(20);
        tree.add(5);
        tree.add(30);

        assertEquals(a, tree.remove(10));

        assertEquals(1, tree.height());
        assertEquals(3, tree.size());
        assertEquals(1, tree.getRoot().getHeight());
        assertEquals(0, tree.getRoot().getBalanceFactor());
        assertEquals((Integer) 20, tree.getRoot().getData());

        AVLNode<Integer> left = tree.getRoot().getLeft();
        assertEquals((Integer) 5, left.getData());
        assertEquals(0, left.getHeight());
        assertEquals(0, left.getBalanceFactor());

        AVLNode<Integer> right = tree.getRoot().getRight();
        assertEquals((Integer) 30, right.getData());
        assertEquals(0, right.getHeight());
        assertEquals(0, right.getBalanceFactor());
    }

    @Test(timeout = TIMEOUT)
    public void removeRightRotate() {
        Integer a = 30;

        tree.add(25);
        tree.add(10);
        tree.add(a);
        tree.add(5);

        assertEquals(a, tree.remove(30));

        assertEquals(1, tree.height());
        assertEquals(3, tree.size());
        assertEquals(1, tree.getRoot().getHeight());
        assertEquals(0, tree.getRoot().getBalanceFactor());
        assertEquals((Integer) 10, tree.getRoot().getData());

        AVLNode<Integer> left = tree.getRoot().getLeft();
        assertEquals((Integer) 5, left.getData());
        assertEquals(0, left.getHeight());
        assertEquals(0, left.getBalanceFactor());

        AVLNode<Integer> right = tree.getRoot().getRight();
        assertEquals((Integer) 25, right.getData());
        assertEquals(0, right.getHeight());
        assertEquals(0, right.getBalanceFactor());
    }

    @Test(timeout = TIMEOUT)
    public void removeLeftRightRotate() {
        Integer a = 20;

        tree.add(10);
        tree.add(a);
        tree.add(5);
        tree.add(7);

        assertEquals(a, tree.remove(20));

        assertEquals(1, tree.height());
        assertEquals(3, tree.size());
        assertEquals(1, tree.getRoot().getHeight());
        assertEquals(0, tree.getRoot().getBalanceFactor());
        assertEquals((Integer) 7, tree.getRoot().getData());

        AVLNode<Integer> left = tree.getRoot().getLeft();
        assertEquals((Integer) 5, left.getData());
        assertEquals(0, left.getHeight());
        assertEquals(0, left.getBalanceFactor());

        AVLNode<Integer> right = tree.getRoot().getRight();
        assertEquals((Integer) 10, right.getData());
        assertEquals(0, right.getHeight());
        assertEquals(0, right.getBalanceFactor());
    }

    @Test(timeout = TIMEOUT)
    public void removeRightLeftRotate() {
        Integer a = 7;

        tree.add(10);
        tree.add(a);
        tree.add(20);
        tree.add(18);

        assertEquals(a, tree.remove(7));

        assertEquals(1, tree.height());
        assertEquals(3, tree.size());
        assertEquals(1, tree.getRoot().getHeight());
        assertEquals(0, tree.getRoot().getBalanceFactor());
        assertEquals((Integer) 18, tree.getRoot().getData());

        AVLNode<Integer> left = tree.getRoot().getLeft();
        assertEquals((Integer) 10, left.getData());
        assertEquals(0, left.getHeight());
        assertEquals(0, left.getBalanceFactor());

        AVLNode<Integer> right = tree.getRoot().getRight();
        assertEquals((Integer) 20, right.getData());
        assertEquals(0, right.getHeight());
        assertEquals(0, right.getBalanceFactor());
    }

    @Test(timeout = TIMEOUT)
    public void removeComplicated1() {
        Integer a = 80;
        Integer b = 100;
        Integer c = 77;

        tree.add(75);
        tree.add(50);
        tree.add(a);
        tree.add(20);
        tree.add(60);
        tree.add(c);
        tree.add(b);

        assertEquals(a, tree.remove(80));
        assertEquals(b, tree.remove(100));
        assertEquals(c, tree.remove(77));

        assertEquals(2, tree.height());
        assertEquals(4, tree.size());
        assertEquals(2, tree.getRoot().getHeight());
        assertEquals(-1, tree.getRoot().getBalanceFactor());
        assertEquals((Integer) 50, tree.getRoot().getData());

        AVLNode<Integer> left = tree.getRoot().getLeft();
        assertEquals((Integer) 20, left.getData());
        assertEquals(0, left.getHeight());
        assertEquals(0, left.getBalanceFactor());

        AVLNode<Integer> right = tree.getRoot().getRight();
        assertEquals((Integer) 75, right.getData());
        assertEquals(1, right.getHeight());
        assertEquals(1, right.getBalanceFactor());

        left = right.getLeft();
        assertEquals((Integer) 60, left.getData());
        assertEquals(0, left.getHeight());
        assertEquals(0, left.getBalanceFactor());
    }

    @Test(timeout = TIMEOUT)
    public void removeComplicated2() {
        Integer a = 75;
        Integer b = 20;
        Integer c = 60;

        tree.add(a);
        tree.add(50);
        tree.add(80);
        tree.add(b);
        tree.add(c);
        tree.add(77);
        tree.add(100);

        assertEquals(a, tree.remove(75));
        assertEquals(b, tree.remove(20));
        assertEquals(c, tree.remove(60));

        assertEquals(2, tree.height());
        assertEquals(4, tree.size());
        assertEquals(2, tree.getRoot().getHeight());
        assertEquals(1, tree.getRoot().getBalanceFactor());
        assertEquals((Integer) 80, tree.getRoot().getData());

        AVLNode<Integer> left = tree.getRoot().getLeft();
        assertEquals((Integer) 50, left.getData());
        assertEquals(1, left.getHeight());
        assertEquals(-1, left.getBalanceFactor());

        AVLNode<Integer> right = tree.getRoot().getRight();
        assertEquals((Integer) 100, right.getData());
        assertEquals(0, right.getHeight());
        assertEquals(0, right.getBalanceFactor());

        right = left.getRight();
        assertEquals((Integer) 77, right.getData());
        assertEquals(0, right.getHeight());
        assertEquals(0, right.getBalanceFactor());
    }

    @Test(timeout = TIMEOUT)
    public void get() {
        Integer a = 40;
        Integer b = 10000;
        Integer c = 30;
        Integer d = 35;

        tree.add(c);
        tree.add(27);
        tree.add(a);
        tree.add(21);
        tree.add(28);
        tree.add(d);
        tree.add(b);

        assertEquals(a, tree.get(40));
        assertEquals(b, tree.get(10000));
        assertEquals(c, tree.get(30));
        assertEquals(d, tree.get(35));
    }

    @Test(timeout = TIMEOUT)
    public void contains() {
        tree.add(30);
        tree.add(27);
        tree.add(40);
        tree.add(21);
        tree.add(28);
        tree.add(35);
        tree.add(10000);

        assertTrue(tree.contains(40));
        assertTrue(tree.contains(35));
        assertTrue(tree.contains(30));
        assertTrue(tree.contains(10000));

        assertFalse(tree.contains(0));
        assertFalse(tree.contains(-30));
        assertFalse(tree.contains(100));
        assertFalse(tree.contains(15));
    }

    @Test(timeout = TIMEOUT)
    public void clear() {
        tree.add(30);
        tree.add(27);
        tree.add(40);
        tree.add(21);
        tree.add(28);
        tree.add(35);
        tree.add(10000);

        tree.clear();
        assertEquals(0, tree.size());
        assertNull(tree.getRoot());
    }

    @Test(timeout = TIMEOUT)
    public void height() {
        assertEquals(-1, tree.height());

        tree.add(30);
        tree.add(27);
        tree.add(40);
        tree.add(21);
        tree.add(28);
        tree.add(35);
        tree.add(10000);

        assertEquals(2, tree.height());

        tree.clear();
        tree.add(10);
        tree.add(1);
        tree.add(15);
        tree.add(2);
        assertEquals(2, tree.height());

        tree.clear();
        tree.add(10);
        tree.add(1);
        assertEquals(1, tree.height());

        tree.clear();
        tree.add(10);
        tree.add(16);
        assertEquals(1, tree.height());
    }

    @Test(timeout = TIMEOUT)
    public void deepestBranches1() {
        tree.add(30);
        tree.add(27);
        tree.add(40);
        tree.add(21);
        tree.add(28);
        tree.add(35);
        tree.add(10000);

        List<Integer> sol = new ArrayList<>(List.of(30, 27, 21, 28, 40, 35, 10000));

        //Should return preorder traversal of all branches since they have same depth
        assertArrayEquals(sol.toArray(), tree.deepestBranches().toArray());
    }

    @Test(timeout = TIMEOUT)
    public void deepestBranches2() {
        tree.add(30);
        tree.add(27);
        tree.add(40);
        tree.add(21);
        tree.add(28);
        tree.add(35);
        tree.add(10000);
        tree.add(29);

        List<Integer> sol = new ArrayList<>(List.of(30, 27, 28, 29));

        //Only one branch of max depth
        assertArrayEquals(sol.toArray(), tree.deepestBranches().toArray());
    }

    @Test(timeout = TIMEOUT)
    public void deepestBranches3() {
        tree.add(30);
        tree.add(27);
        tree.add(40);
        tree.add(21);
        tree.add(28);
        tree.add(35);
        tree.add(10000);
        tree.add(29);
        tree.add(37);
        tree.add(9999);

        List<Integer> sol = new ArrayList<>(List.of(30, 27, 28, 29, 40, 35, 37, 10000, 9999));

        //3 branches of max depth
        assertArrayEquals(sol.toArray(), tree.deepestBranches().toArray());
    }

    @Test(timeout = TIMEOUT)
    public void sortedInBetween1() {
        tree.add(30);
        tree.add(27);
        tree.add(40);
        tree.add(21);
        tree.add(28);
        tree.add(35);
        tree.add(10000);
        tree.add(29);
        tree.add(37);
        tree.add(9999);

        List<Integer> sol = new ArrayList<>(List.of(21, 27, 28, 29, 30, 35, 37, 40, 9999, 10000));

        //Should include all elements
        assertArrayEquals(sol.toArray(), tree.sortedInBetween(20, 10001).toArray());
    }

    @Test(timeout = TIMEOUT)
    public void sortedInBetween2() {
        tree.add(30);
        tree.add(27);
        tree.add(40);
        tree.add(21);
        tree.add(28);
        tree.add(35);
        tree.add(10000);
        tree.add(29);
        tree.add(37);
        tree.add(9999);

        List<Integer> sol = new ArrayList<>(List.of(28, 29, 30, 35, 37));

        assertArrayEquals(sol.toArray(), tree.sortedInBetween(27, 40).toArray());
    }

    @Test(timeout = TIMEOUT)
    public void sortedInBetween3() {
        tree.add(30);
        tree.add(27);
        tree.add(40);
        tree.add(21);
        tree.add(28);
        tree.add(35);
        tree.add(10000);
        tree.add(29);
        tree.add(37);
        tree.add(9999);

        List<Integer> sol = new ArrayList<>(List.of(9999, 10000));

        assertArrayEquals(sol.toArray(), tree.sortedInBetween(9998, 100000).toArray());
    }

    @Test(timeout = TIMEOUT)
    public void sortedInBetween4() {
        tree.add(30);
        tree.add(27);
        tree.add(40);
        tree.add(21);
        tree.add(28);
        tree.add(35);
        tree.add(10000);
        tree.add(29);
        tree.add(37);
        tree.add(9999);

        List<Integer> sol = new ArrayList<>();

        //Should be empty
        assertArrayEquals(sol.toArray(), tree.sortedInBetween(-50, 0).toArray());
        assertArrayEquals(sol.toArray(), tree.sortedInBetween(30, 30).toArray());
    }

    @Test(timeout = TIMEOUT)
    public void sortedInBetween5() {
        tree.add(30);
        tree.add(27);
        tree.add(40);
        tree.add(21);
        tree.add(28);
        tree.add(35);
        tree.add(10000);
        tree.add(29);
        tree.add(37);
        tree.add(9999);

        List<Integer> sol = new ArrayList<>(List.of(21, 27, 28, 29));

        assertArrayEquals(sol.toArray(), tree.sortedInBetween(20, 30).toArray());
    }
}
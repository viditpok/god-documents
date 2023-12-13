import org.junit.Before;
import org.junit.Test;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Pattern;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.assertArrayEquals;

/**
 * Tests to see if Galil Rule is implemented properly (if attempted)
 *
 * @author Abhinav Vemulapalli
 * @version 1.0
 */
public class GalilRuleTests {
    private static final int TIMEOUT = 200;

    private String pattern;

    private String text;

    private List<Integer> answer;

    private CharacterComparator comparator;

    @Before
    public void setUp() {
        answer = new ArrayList<>();
        comparator = new CharacterComparator();
    }

    @Test(timeout = TIMEOUT)
    public void testGalilExample1() {
        text = "abcdeabcdabcdeabcdabcdeabcd";
        pattern = "abcdeabcd";

        answer.add(0);
        answer.add(9);
        answer.add(18);

        assertEquals(answer, PatternMatching.boyerMooreGalilRule(pattern, text, comparator));
        assertTrue("Did not use comparator.", comparator.getComparisonCount() != 0);
        assertEquals("Comparison count was " + comparator.getComparisonCount() + ". Should be 37 (29 from Galil Rule and 8 from building failure table).", 37, comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testGalilLongerPattern() {
        text = "idk";
        pattern = "longer";

        assertEquals(new ArrayList<Integer>(), PatternMatching.boyerMooreGalilRule(pattern, text, comparator));
        assertEquals(0, comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testGalilExample2() {
        text = "laburbujabuburburburbu";
        pattern = "burbu";

        answer.add(2);
        answer.add(11);
        answer.add(14);
        answer.add(17);

        assertEquals(answer, PatternMatching.boyerMooreGalilRule(pattern, text, comparator));
        assertTrue("Did not use comparator.", comparator.getComparisonCount() != 0);
        assertEquals("Comparison count was " + comparator.getComparisonCount() + ". Should be 26 (22 from Galil Rule and 4 from building failure table).", 26, comparator.getComparisonCount());

    }

    @Test(timeout = TIMEOUT)
    public void testGalilExample3() {
        text = "ababaababcababcabbababc";
        pattern = "ababc";

        answer.add(5);
        answer.add(10);
        answer.add(18);

        assertEquals(answer, PatternMatching.boyerMooreGalilRule(pattern, text, comparator));
        assertTrue("Did not use comparator.", comparator.getComparisonCount() != 0);
        assertEquals("Comparison count was " + comparator.getComparisonCount() + ". Should be 25 (20 from Galil Rule and 5 from building failure table).", 25, comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void superSimpleTest() {
        text = "oggogoooggoog";
        pattern = "go";

        answer.add(2);
        answer.add(4);
        answer.add(9);

        assertEquals(answer, PatternMatching.boyerMooreGalilRule(pattern, text, comparator));
        assertTrue("Did not use comparator.", comparator.getComparisonCount() != 0);
        assertEquals("Comparison count was " + comparator.getComparisonCount() + ". Should be 14 (13 from Galil Rule and 1 from building failure table).", 14, comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void comingUpWithTestsIsHard() {
        text = "bababababab";
        pattern = "ababa";

        answer.add(1);
        answer.add(3);
        answer.add(5);

        assertEquals(answer, PatternMatching.boyerMooreGalilRule(pattern, text, comparator));
        assertTrue("Did not use comparator.", comparator.getComparisonCount() != 0);
        assertEquals("Comparison count was " + comparator.getComparisonCount() + ". Should be 14 (10 from Galil Rule and 4 from building failure table).", 14, comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void iHopeTheseAreEnoughTestCases() {
        text = "ANPANMAN";
        pattern = "PAN";

        answer.add(2);
        assertEquals(answer, PatternMatching.boyerMooreGalilRule(pattern, text, comparator));
        assertTrue("Did not use comparator.", comparator.getComparisonCount() != 0);
        assertEquals("Comparison count was " + comparator.getComparisonCount() + ". Should be 9 (6 from Galil Rule and 3 from building failure table).", 9, comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void notALotOfPeopleAreGoingToDoGalilRuleAnywaysHopeMyTestsAreGood() {
        text = "AABAACAADAABAABA";
        pattern = "AABA";

        answer.add(0);
        answer.add(9);
        answer.add(12);
        assertEquals(answer, PatternMatching.boyerMooreGalilRule(pattern, text, comparator));
        assertTrue("Did not use comparator.", comparator.getComparisonCount() != 0);
        assertEquals("Comparison count was " + comparator.getComparisonCount() + ". Should be 19 (15 from Galil Rule and 4 from building failure table).", 19, comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void iForgotCSVisExistedButNowImTooLazy() {
        text = "racecaracecareracedfa";
        pattern = "racecar";

        answer.add(0);
        answer.add(6);
        assertEquals(answer, PatternMatching.boyerMooreGalilRule(pattern, text, comparator));
        assertTrue("Did not use comparator.", comparator.getComparisonCount() != 0);
        assertEquals("Comparison count was " + comparator.getComparisonCount() + ". Should be 20 (14 from Galil Rule and 6 from building failure table).", 20, comparator.getComparisonCount());
    }
}
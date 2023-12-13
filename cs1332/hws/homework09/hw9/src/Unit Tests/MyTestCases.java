import org.junit.Before;
import org.junit.Test;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import static org.junit.Assert.*;
import static org.junit.Assert.assertThrows;

public class MyTestCases {
    private static final int TIMEOUT = 200;
    private CharacterComparator comparator;
    private String patternLonger;
    private String patternLongerText;
    private List<Integer> patternLongerResult;
    private String noMatchPattern;
    private String noMatchText;
    private List<Integer> noMatchResult;
    private String multiMatchPattern;
    private String multiMatchText;
    private List<Integer> multiMatchResult;
    private String everythingMatchesPattern;
    private String everythingMatchesText;
    private List<Integer> everythingMatchesResult;
    private String multiMatchPattern2;
    private String multiMatchText2;
    private List<Integer> multiMatchResult2;
    private String multiMatchPattern3;
    private String multiMatchText3;
    private List<Integer> multiMatchResult3;
    private String samePattern;
    private List<Integer> sameResult;
    private String randomPattern;
    private String randomText;
    private List<Integer> randomResult;

    @Before
    public void setUp() {
        noMatchPattern = "pattern";
        noMatchText = "text with no match";
        noMatchResult = new ArrayList<>();

        samePattern = "Hello World!";
        sameResult = new ArrayList<>();
        sameResult.add(0);

        patternLonger = "pattern";
        patternLongerText = "text";
        patternLongerResult = new ArrayList<>();

        multiMatchPattern = "asdf";
        multiMatchText = "asdfqwreinbpasdfqweniasdf";
        multiMatchResult = new ArrayList<>();
        multiMatchResult.add(0);
        multiMatchResult.add(12);
        multiMatchResult.add(21);

        multiMatchPattern2 = "asdf";
        multiMatchText2 = "rpoinasdfirbnundjkasdfeuirnpvoiunasdfpqouernasdf";
        multiMatchResult2 = new ArrayList<>();
        multiMatchResult2.add(5);
        multiMatchResult2.add(18);
        multiMatchResult2.add(33);
        multiMatchResult2.add(44);

        multiMatchPattern3 = "001";
        multiMatchText3 = "10000001000000100001100001";
        multiMatchResult3 = new ArrayList<>();
        multiMatchResult3.add(5);
        multiMatchResult3.add(12);
        multiMatchResult3.add(17);
        multiMatchResult3.add(23);

        everythingMatchesPattern = "101";
        everythingMatchesText = "1010101010101010101010101";
        everythingMatchesResult = new ArrayList<>();
        for (int i = 0; i < 12; i++) {
            everythingMatchesResult.add(i * 2);
        }

        randomPattern = "1123453145132413";
        randomText = "";
        randomResult = new ArrayList<>();
        Random rn = new Random();
        for (int i = 0; i < 201; i++) {
            if (i % 5 == 0) {
                randomText += randomPattern;
                randomResult.add(i * 8);
            } else {
                for (int j = 0; j < 6; j++) {
                    randomText += Integer.toString(rn.nextInt(6));
                }
            }
        }

        comparator = new CharacterComparator();
    }

    @Test(timeout = TIMEOUT)
    public void testKmpException() {
        assertThrows(IllegalArgumentException.class, () -> {
            PatternMatching.kmp(null, "qwerty", comparator);
        });
    }

    @Test(timeout = TIMEOUT)
    public void testKmpException2() {
        assertThrows(IllegalArgumentException.class, () -> {
            PatternMatching.kmp("", "qwerty", comparator);
        });
    }

    @Test(timeout = TIMEOUT)
    public void testKmpException3() {
        assertThrows(IllegalArgumentException.class, () -> {
            PatternMatching.kmp("qwerty", null, comparator);
        });
    }

    @Test(timeout = TIMEOUT)
    public void testKmpException4() {
        assertThrows(IllegalArgumentException.class, () -> {
            PatternMatching.kmp("asdf", "qwerty", null);
        });
    }

    @Test(timeout = TIMEOUT)
    public void testBuildFailureTableException() {
        assertThrows(IllegalArgumentException.class, () -> {
            PatternMatching.buildFailureTable(null, comparator);
        });
    }

    @Test(timeout = TIMEOUT)
    public void testBuildFailureTableException2() {
        assertThrows(IllegalArgumentException.class, () -> {
            PatternMatching.buildFailureTable("asdf", null);
        });
    }

    @Test(timeout = TIMEOUT)
    public void testBuildFailureTable1() {
        int[] failureTable = PatternMatching
                .buildFailureTable("12332334123", comparator);
        int[] expected = {0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3};
        assertArrayEquals(expected, failureTable);
        assertTrue("Comparator was not used...",
                comparator.getComparisonCount() != 0);
        assertEquals("There should be 10 comparisons, instead there were actually: "
                + comparator.getComparisonCount(), 10, comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBuildFailureTable2() {
        int[] failureTable = PatternMatching
                .buildFailureTable("aaaaaaaab", comparator);
        int[] expected = {0, 1, 2, 3, 4, 5, 6, 7, 0};
        assertArrayEquals(expected, failureTable);
        assertTrue("Comparator was not used...",
                comparator.getComparisonCount() != 0);
        assertEquals("There should be 15 comparisons, instead there were actually: "
                + comparator.getComparisonCount(), 15, comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBuildFailureTable3() {
        int[] failureTable = PatternMatching
                .buildFailureTable("343434134343", comparator);
        int[] expected = {0, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5};
        assertArrayEquals(expected, failureTable);
        assertTrue("Comparator was not used...",
                comparator.getComparisonCount() != 0);
        assertEquals("There should be 13 comparisons, instead there were actually: "
                + comparator.getComparisonCount(), 13, comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBuildFailureTableEmpty() {
        int[] failureTable = PatternMatching
                .buildFailureTable("", comparator);
        assertArrayEquals(new int[0], failureTable);
        assertEquals("There should be 0 comparisons, instead there were actually: "
                + comparator.getComparisonCount(), 0, comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testKmpPatternLonger() {
        assertEquals(patternLongerResult,
                PatternMatching.kmp(patternLonger, patternLongerText, comparator));
        assertEquals("There should be 0 comparisons, instead there were actually: "
                + comparator.getComparisonCount(), 0, comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testKmpNoMatch() {
        assertEquals(noMatchResult,
                PatternMatching.kmp(noMatchPattern, noMatchText, comparator));
        assertTrue("Comparator was not used...",
                comparator.getComparisonCount() != 0);
        assertEquals("There should be 18 comparisons, instead there were actually: "
                + comparator.getComparisonCount(), 18, comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testKmpSamePattern() {
        assertEquals(sameResult,
                PatternMatching.kmp(samePattern, samePattern, comparator));
        assertTrue("Comparator was not used...",
                comparator.getComparisonCount() != 0);
        assertEquals("There should be 23 comparisons, instead there were actually: "
                + comparator.getComparisonCount(), 23, comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testKmpMultiMatchPattern() {
        assertEquals(multiMatchResult,
                PatternMatching.kmp(multiMatchPattern, multiMatchText, comparator));
        assertTrue("Comparator was not used...",
                comparator.getComparisonCount() != 0);
        assertEquals("There should be 28 comparisons, instead there were actually: "
                + comparator.getComparisonCount(), 28, comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testKmpEverythingMatches() {
        assertEquals(everythingMatchesResult,
                PatternMatching.kmp(everythingMatchesPattern, everythingMatchesText, comparator));
        assertTrue("Comparator was not used...",
                comparator.getComparisonCount() != 0);
        assertEquals("There should be 27 comparisons, instead there were actually: "
                + comparator.getComparisonCount(), 27, comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testKmpMultiMatchPattern2() {
        assertEquals(multiMatchResult2,
                PatternMatching.kmp(multiMatchPattern2, multiMatchText2, comparator));
        assertTrue("Comparator was not used...",
                comparator.getComparisonCount() != 0);
        assertEquals("There should be 51 comparisons, instead there were actually: "
                + comparator.getComparisonCount(), 51, comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testKmpMultiMatchPattern3() {
        assertEquals(multiMatchResult3,
                PatternMatching.kmp(multiMatchPattern3, multiMatchText3, comparator));
        assertTrue("Comparator was not used...",
                comparator.getComparisonCount() != 0);
        assertEquals("There should be 41 comparisons, instead there were actually: "
                + comparator.getComparisonCount(), 41, comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testKmpRandomPattern() {
        assertEquals(randomResult,
                PatternMatching.kmp(randomPattern, randomText, comparator));
        assertTrue("Comparator was not used...",
                comparator.getComparisonCount() != 0);
    }

    @Test(timeout = TIMEOUT)
    public void testBoyerMooreException() {
        assertThrows(IllegalArgumentException.class, () -> {
            PatternMatching.boyerMoore(null, "text", comparator);
        });
    }

    @Test(timeout = TIMEOUT)
    public void testBoyerMooreException2() {
        assertThrows(IllegalArgumentException.class, () -> {
            PatternMatching.boyerMoore("", "text", comparator);
        });
    }

    @Test(timeout = TIMEOUT)
    public void testBoyerMooreException3() {
        assertThrows(IllegalArgumentException.class, () -> {
            PatternMatching.boyerMoore("pattern", null, comparator);
        });
    }

    @Test(timeout = TIMEOUT)
    public void testBoyerMooreException4() {
        assertThrows(IllegalArgumentException.class, () -> {
            PatternMatching.boyerMoore("match", "could match", null);
        });
    }

    @Test(timeout = TIMEOUT)
    public void testBuildLastTableException() {
        assertThrows(IllegalArgumentException.class, () -> {
            PatternMatching.buildLastTable(null);
        });
    }

    @Test(timeout = TIMEOUT)
    public void testBuildLastTable1() {
        Map<Character, Integer> lastTable = PatternMatching
                .buildLastTable("2354152355215345525345534");
        Map<Character, Integer> expectedLastTable = new HashMap<>();
        expectedLastTable.put('1', 11);
        expectedLastTable.put('2', 17);
        expectedLastTable.put('3', 23);
        expectedLastTable.put('4', 24);
        expectedLastTable.put('5', 22);
        assertEquals(expectedLastTable, lastTable);
    }

    @Test(timeout = TIMEOUT)
    public void testBuildLastTable2() {
        Map<Character, Integer> lastTable = PatternMatching
                .buildLastTable("101010");
        Map<Character, Integer> expectedLastTable = new HashMap<>();
        expectedLastTable.put('1', 4);
        expectedLastTable.put('0', 5);
        assertEquals(expectedLastTable, lastTable);
    }

    @Test(timeout = TIMEOUT)
    public void testBuildLastTable3() {
        Map<Character, Integer> lastTable = PatternMatching
                .buildLastTable("1234567");
        Map<Character, Integer> expectedLastTable = new HashMap<>();
        expectedLastTable.put('1', 0);
        expectedLastTable.put('2', 1);
        expectedLastTable.put('3', 2);
        expectedLastTable.put('4', 3);
        expectedLastTable.put('5', 4);
        expectedLastTable.put('6', 5);
        expectedLastTable.put('7', 6);
        assertEquals(expectedLastTable, lastTable);
    }

    @Test(timeout = TIMEOUT)
    public void testBuildLastTableEmpty() {
        Map<Character, Integer> lastTable = PatternMatching
                .buildLastTable("");
        Map<Character, Integer> expectedLastTable = new HashMap<>();
        assertEquals(expectedLastTable, lastTable);
    }

    @Test(timeout = TIMEOUT)
    public void testBoyerMoorePatternLonger() {
        assertEquals(patternLongerResult,
                PatternMatching.boyerMoore(patternLonger, patternLongerText, comparator));
        assertEquals("There should be 0 comparisons, instead there were actually: "
                + comparator.getComparisonCount(), 0, comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBoyerMooreNoMatch() {
        assertEquals(noMatchResult,
                PatternMatching.boyerMoore(noMatchPattern, noMatchText, comparator));
        assertTrue("Comparator was not used...",
                comparator.getComparisonCount() != 0);
        assertEquals("There should be 2 comparisons, instead there were actually: "
                + comparator.getComparisonCount(), 2, comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBoyerMooreSamePattern() {
        assertEquals(sameResult,
                PatternMatching.boyerMoore(samePattern, samePattern, comparator));
        assertTrue("Comparator was not used...",
                comparator.getComparisonCount() != 0);
        assertEquals("There should be 12 comparisons, instead there were actually: "
                + comparator.getComparisonCount(), 12, comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBoyerMooreMultiMatchPattern() {
        assertEquals(multiMatchResult,
                PatternMatching.boyerMoore(multiMatchPattern, multiMatchText, comparator));
        assertTrue("Comparator was not used...",
                comparator.getComparisonCount() != 0);
        assertEquals("There should be 17 comparisons, instead there were actually: "
                + comparator.getComparisonCount(), 17, comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBoyerMooreMultiMatchPattern2() {
        assertEquals(multiMatchResult2,
                PatternMatching.boyerMoore(multiMatchPattern2, multiMatchText2, comparator));
        assertTrue("Comparator was not used...",
                comparator.getComparisonCount() != 0);
        assertEquals("There should be 28 comparisons, instead there were actually: "
                + comparator.getComparisonCount(), 28, comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBoyerMooreMultiMatchPattern3() {
        assertEquals(multiMatchResult3,
                PatternMatching.boyerMoore(multiMatchPattern3, multiMatchText3, comparator));
        assertTrue("Comparator was not used...",
                comparator.getComparisonCount() != 0);
        assertEquals("There should be 33 comparisons, instead there were actually: "
                + comparator.getComparisonCount(), 33, comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBoyerMooreAllMatch() {
        assertEquals(everythingMatchesResult,
                PatternMatching.boyerMoore(everythingMatchesPattern, everythingMatchesText, comparator));
        assertTrue("Comparator was not used...",
                comparator.getComparisonCount() != 0);
        assertEquals("There should be 47 comparisons, instead there were actually: "
                + comparator.getComparisonCount(), 47, comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBoyerMooreRandomPattern() {
        assertEquals(randomResult,
                PatternMatching.boyerMoore(randomPattern, randomText, comparator));
        assertTrue("Comparator was not used...",
                comparator.getComparisonCount() != 0);
    }

    @Test(timeout = TIMEOUT)
    public void testRabinKarpException() {
        assertThrows(IllegalArgumentException.class, () -> {
            PatternMatching.rabinKarp(null, "text", comparator);
        });
    }

    @Test(timeout = TIMEOUT)
    public void testRabinKarpException2() {
        assertThrows(IllegalArgumentException.class, () -> {
            PatternMatching.rabinKarp("", "text", comparator);
        });
    }

    @Test(timeout = TIMEOUT)
    public void testRabinKarpException3() {
        assertThrows(IllegalArgumentException.class, () -> {
            PatternMatching.rabinKarp("pattern", null, comparator);
        });
    }

    @Test(timeout = TIMEOUT)
    public void testRabinKarpException4() {
        assertThrows(IllegalArgumentException.class, () -> {
            PatternMatching.rabinKarp("match", "could match", null);
        });
    }

    @Test(timeout = TIMEOUT)
    public void testRabinKarpPatternLonger() {
        assertEquals(patternLongerResult,
                PatternMatching.rabinKarp(patternLonger, patternLongerText, comparator));
        assertEquals("There should be 0 comparisons, instead there were actually: "
                + comparator.getComparisonCount(), 0, comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testRabinKarpNoMatch() {
        assertEquals(noMatchResult,
                PatternMatching.rabinKarp(noMatchPattern, noMatchText, comparator));
        assertEquals("There should be 0 comparisons, instead there were actually: "
                + comparator.getComparisonCount(), 0, comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testRabinKarpSamePattern() {
        assertEquals(sameResult,
                PatternMatching.rabinKarp(samePattern, samePattern, comparator));
        assertTrue("Comparator was not used...",
                comparator.getComparisonCount() != 0);
        assertEquals("There should be 12 comparisons, instead there were actually: "
                + comparator.getComparisonCount(), 12, comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testRabinKarpMultiMatchPattern() {
        assertEquals(multiMatchResult,
                PatternMatching.rabinKarp(multiMatchPattern, multiMatchText, comparator));
        assertTrue("Comparator was not used...",
                comparator.getComparisonCount() != 0);
        assertEquals("There should be 12 comparisons, instead there were actually: "
                + comparator.getComparisonCount(), 12, comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testRabinKarpMultiMatchPattern2() {
        assertEquals(multiMatchResult2,
                PatternMatching.rabinKarp(multiMatchPattern2, multiMatchText2, comparator));
        assertTrue("Comparator was not used...",
                comparator.getComparisonCount() != 0);
        assertEquals("There should be 16 comparisons, instead there were actually: "
                + comparator.getComparisonCount(), 16, comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testRabinKarpMultiMatchPattern3() {
        assertEquals(multiMatchResult3,
                PatternMatching.rabinKarp(multiMatchPattern3, multiMatchText3, comparator));
        assertTrue("Comparator was not used...",
                comparator.getComparisonCount() != 0);
        assertEquals("There should be 12 comparisons, instead there were actually: "
                + comparator.getComparisonCount(), 12, comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testRabinKarpAllMatch() {
        assertEquals(everythingMatchesResult,
                PatternMatching.rabinKarp(everythingMatchesPattern, everythingMatchesText, comparator));
        assertTrue("Comparator was not used...",
                comparator.getComparisonCount() != 0);
        assertEquals("There should be 36 comparisons, instead there were actually: "
                + comparator.getComparisonCount(), 36, comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testRabinKarpRandomPattern() {
        assertEquals(randomResult,
                PatternMatching.rabinKarp(randomPattern, randomText, comparator));
        assertTrue("Comparator was not used...",
                comparator.getComparisonCount() != 0);
    }
}
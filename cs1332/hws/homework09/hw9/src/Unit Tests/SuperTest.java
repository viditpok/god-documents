import org.junit.Before;
import org.junit.Test;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import static org.junit.Assert.*;
import static org.junit.Assert.assertThrows;

/**
 * @author Rishi Soni
 * @version 1.0
 */
public class SuperTest {

    private static final int TIMEOUT = 350;

    private String longPattern;
    private String longPatternText;
    private List<Integer> longPatternAnswer;
    private String noMatchPattern;
    private String noMatchPatternText;
    private List<Integer> noMatchPatternAnswer;
    private String multiPattern1;
    private String multiPattern1Text;
    private List<Integer> multiplePattern1Answer;
    private String allMatchPattern;
    private String allMatchPatternText;
    private List<Integer> allMatchPatternAnswer;
    private CharacterComparator comparator;
    private String multiPattern2;
    private String multiPattern2Text;
    private List<Integer> multiplePattern2Answer;
    private String multiPattern3;
    private String multiPattern3Text;
    private List<Integer> multiplePattern3Answer;
    private String samePattern;
    private List<Integer> samePatternAnswer;
    private String extremePattern1;
    private String extremePattern1Text;
    private List<Integer> extremePattern1Answer;
    private String extremePattern2;
    private String extremePattern2Text;
    private List<Integer> extremePattern2Answer;


    @Before
    public void setUp() {
        longPattern = "will not match";
        longPatternText = "12345";
        longPatternAnswer = new ArrayList<>();

        noMatchPattern = "abcdef";
        noMatchPatternText = "abcdehabcaaabcfdef";
        noMatchPatternAnswer = new ArrayList<>();

        samePattern = "aeaoahol";
        samePatternAnswer = new ArrayList<>();
        samePatternAnswer.add(0);

        multiPattern1 = "12312";
        multiPattern1Text = "1234512312331212312451312312";
        multiplePattern1Answer = new ArrayList<>();
        multiplePattern1Answer.add(5);
        multiplePattern1Answer.add(14);
        multiplePattern1Answer.add(23);

        allMatchPattern = "aba";
        allMatchPatternText = "ababababababababababababa";
        allMatchPatternAnswer = new ArrayList<>();
        for (int i = 0; i < 12; i++) {
            allMatchPatternAnswer.add(i * 2);
        }

        multiPattern2 = "1343412";
        multiPattern2Text = "1343412012134420134341211340341134341221340134341293413134";
        multiplePattern2Answer = new ArrayList<>();
        multiplePattern2Answer.add(0);
        multiplePattern2Answer.add(16);
        multiplePattern2Answer.add(31);
        multiplePattern2Answer.add(43);

        multiPattern3 = "aaaah";
        multiPattern3Text = "haaaaaaaaaaaaaaaaaaahaaaaaaaaaaaaaaaaaaaah";
        multiplePattern3Answer = new ArrayList<>();
        multiplePattern3Answer.add(16);
        multiplePattern3Answer.add(37);

        extremePattern1 = "1123453145132413";
        extremePattern1Text = "";
        extremePattern1Answer = new ArrayList<>();
        Random rn = new Random();
        for (int i = 0; i < 501; i++) {
            if (i % 5 == 0) {
                extremePattern1Text += extremePattern1;
                extremePattern1Answer.add(i * 8);
            } else {
                for (int j = 0; j < 6; j++) {
                    extremePattern1Text += Integer.toString(rn.nextInt(6));
                }
            }
        }

        extremePattern2 = "aababcabcdabcdeabcdef";
        extremePattern2Text = "";
        extremePattern2Answer = new ArrayList<>();
        for (int i = 0; i < 201; i++) {
            if (i % 4 == 0) {
                extremePattern2Text += extremePattern2;
                extremePattern2Answer.add(i * 12);
            } else {
                for (int j = 0; j < 9; j++) {
                    extremePattern2Text += (char) (rn.nextInt(5) + 97);
                }
            }
        }

        comparator = new CharacterComparator();
    }

    @Test(timeout = TIMEOUT)
    public void kmpExceptions() {
        assertThrows(IllegalArgumentException.class, () -> {
            PatternMatching.kmp(null, "abc", comparator);
        });

        assertThrows(IllegalArgumentException.class, () -> {
            PatternMatching.kmp("", "abc", comparator);
        });

        assertThrows(IllegalArgumentException.class, () -> {
            PatternMatching.kmp("a", null, comparator);
        });

        assertThrows(IllegalArgumentException.class, () -> {
            PatternMatching.kmp("a", "abc", null);
        });
    }

    @Test(timeout = TIMEOUT)
    public void boyerMooreExceptions() {
        assertThrows(IllegalArgumentException.class, () -> {
            PatternMatching.boyerMoore(null, "abc", comparator);
        });

        assertThrows(IllegalArgumentException.class, () -> {
            PatternMatching.boyerMoore("", "abc", comparator);
        });

        assertThrows(IllegalArgumentException.class, () -> {
            PatternMatching.boyerMoore("a", null, comparator);
        });

        assertThrows(IllegalArgumentException.class, () -> {
            PatternMatching.boyerMoore("a", "abc", null);
        });
    }

    @Test(timeout = TIMEOUT)
    public void rabinKarpExceptions() {
        assertThrows(IllegalArgumentException.class, () -> {
            PatternMatching.rabinKarp(null, "abc", comparator);
        });

        assertThrows(IllegalArgumentException.class, () -> {
            PatternMatching.rabinKarp("", "abc", comparator);
        });

        assertThrows(IllegalArgumentException.class, () -> {
            PatternMatching.rabinKarp("a", null, comparator);
        });

        assertThrows(IllegalArgumentException.class, () -> {
            PatternMatching.rabinKarp("a", "abc", null);
        });
    }

    @Test(timeout = TIMEOUT)
    public void buildFailureTableExceptions() {
        assertThrows(IllegalArgumentException.class, () -> {
            PatternMatching.buildFailureTable(null, comparator);
        });

        assertThrows(IllegalArgumentException.class, () -> {
            PatternMatching.buildFailureTable("abc", null);
        });
    }

    @Test(timeout = TIMEOUT)
    public void buildLastTableExceptions() {
        assertThrows(IllegalArgumentException.class, () -> {
            PatternMatching.buildLastTable(null);
        });
    }

    @Test(timeout = TIMEOUT)
    public void buildFailureTable1() {
        int[] failureTable = PatternMatching
                .buildFailureTable("aabcdaabdaabcdb", comparator);
        int[] expected = {0, 1, 0, 0, 0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 0};
        assertArrayEquals(expected, failureTable);
        assertTrue("Did not use the comparator.",
                comparator.getComparisonCount() != 0);
        assertEquals("Comparison count was " + comparator.getComparisonCount()
                + ". Should be 17.", 17, comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void buildFailureTable2() {
        int[] failureTable = PatternMatching
                .buildFailureTable("11111111110", comparator);
        int[] expected = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0};
        assertArrayEquals(expected, failureTable);
        assertTrue("Did not use the comparator.",
                comparator.getComparisonCount() != 0);
        assertEquals("Comparison count was " + comparator.getComparisonCount()
                + ". Should be 19.", 19, comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void buildFailureTable3() {
        int[] failureTable = PatternMatching
                .buildFailureTable("abcdefbcdabcab", comparator);
        int[] expected = {0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 1, 2};
        assertArrayEquals(expected, failureTable);
        assertTrue("Did not use the comparator.",
                comparator.getComparisonCount() != 0);
        assertEquals("Comparison count was " + comparator.getComparisonCount()
                + ". Should be 14.", 14, comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void buildFailureTable4() {
        int[] failureTable = PatternMatching
                .buildFailureTable("", comparator);
        assertArrayEquals(new int[0], failureTable);
        assertEquals("Comparison count was " + comparator.getComparisonCount()
                + ". Should be 0.", 0, comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void buildLastTable1() {
        Map<Character, Integer> lastTable = PatternMatching
                .buildLastTable("abcdabcccccccccde");
        Map<Character, Integer> expectedLastTable = new HashMap<>();
        expectedLastTable.put('a', 4);
        expectedLastTable.put('b', 5);
        expectedLastTable.put('c', 14);
        expectedLastTable.put('d', 15);
        expectedLastTable.put('e', 16);
        assertEquals(expectedLastTable, lastTable);
    }

    @Test(timeout = TIMEOUT)
    public void buildLastTable2() {
        Map<Character, Integer> lastTable = PatternMatching
                .buildLastTable("");
        Map<Character, Integer> expectedLastTable = new HashMap<>();
        assertEquals(expectedLastTable, lastTable);
    }

    @Test(timeout = TIMEOUT)
    public void buildLastTable3() {
        Map<Character, Integer> lastTable = PatternMatching
                .buildLastTable("1010101011010101010101");
        Map<Character, Integer> expectedLastTable = new HashMap<>();
        expectedLastTable.put('1', 21);
        expectedLastTable.put('0', 20);
        assertEquals(expectedLastTable, lastTable);
    }

    @Test(timeout = TIMEOUT)
    public void buildLastTable4() {
        Map<Character, Integer> lastTable = PatternMatching
                .buildLastTable("abcdefabcdef");
        Map<Character, Integer> expectedLastTable = new HashMap<>();
        expectedLastTable.put('a', 6);
        expectedLastTable.put('b', 7);
        expectedLastTable.put('c', 8);
        expectedLastTable.put('d', 9);
        expectedLastTable.put('e', 10);
        expectedLastTable.put('f', 11);
        assertEquals(expectedLastTable, lastTable);
    }

    @Test(timeout = TIMEOUT)
    public void kmpNoMatch() {
        /*
            pattern: abcdef
            text: abcdehabcaaabcfdef
            indices: -
            expected total comparison: 24
         */
        assertEquals(noMatchPatternAnswer,
                PatternMatching.kmp(noMatchPattern, noMatchPatternText, comparator));
        assertTrue("Did not use the comparator.",
                comparator.getComparisonCount() != 0);
        assertEquals("Comparison count was " + comparator.getComparisonCount()
                + ". Should be 24.", 24, comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void boyerMooreNoMatch() {
        /*
            pattern: abcdef
            text: abcdehabcaaabcfdef
            indices: -
            expected total comparison: 7
         */
        assertEquals(noMatchPatternAnswer,
                PatternMatching.boyerMoore(noMatchPattern, noMatchPatternText, comparator));
        assertTrue("Did not use the comparator.",
                comparator.getComparisonCount() != 0);
        assertEquals("Comparison count was " + comparator.getComparisonCount()
                + ". Should be 7.", 7, comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void rabinKarpNoMatch() {
        /*
            pattern: abcdef
            text: abcdehabcaaabcfdef
            indices: -
            expected total comparison: 0
         */
        assertEquals(noMatchPatternAnswer,
                PatternMatching.rabinKarp(noMatchPattern, noMatchPatternText, comparator));
        assertEquals("Comparison count was " + comparator.getComparisonCount()
                + ". Should be 0.", 0, comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void kmpLongPattern() {
        /*
            pattern: will not match
            text: 12345
            indices: -
            expected total comparison: 0
         */
        assertEquals(longPatternAnswer,
                PatternMatching.kmp(longPattern, longPatternText, comparator));
        assertEquals("Comparison count was " + comparator.getComparisonCount()
                + ". Should be 0.", 0, comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void boyerMooreLongPattern() {
        /*
            pattern: will not match
            text: 12345
            indices: -
            expected total comparison: 0
         */
        assertEquals(longPatternAnswer,
                PatternMatching.boyerMoore(longPattern, longPatternText, comparator));
        assertEquals("Comparison count was " + comparator.getComparisonCount()
                + ". Should be 0.", 0, comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void rabinKarpLongPattern() {
        /*
            pattern: will not match
            text: 12345
            indices: -
            expected total comparison: 0
         */
        assertEquals(longPatternAnswer,
                PatternMatching.rabinKarp(longPattern, longPatternText, comparator));
        assertEquals("Comparison count was " + comparator.getComparisonCount()
                + ". Should be 0.", 0, comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void kmpSamePattern() {
        /*
            pattern: aeaoahol
            text: aeaoahol
            indices: 0
            expected total comparison: 17
         */
        assertEquals(samePatternAnswer,
                PatternMatching.kmp(samePattern, samePattern, comparator));
        assertTrue("Did not use the comparator.",
                comparator.getComparisonCount() != 0);
        assertEquals("Comparison count was " + comparator.getComparisonCount()
                + ". Should be 17.", 17, comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void boyerMooreSamePattern() {
        /*
            pattern: aeaoahol
            text: aeaoahol
            indices: 0
            expected total comparison: 8
         */
        assertEquals(samePatternAnswer,
                PatternMatching.boyerMoore(samePattern, samePattern, comparator));
        assertTrue("Did not use the comparator.",
                comparator.getComparisonCount() != 0);
        assertEquals("Comparison count was " + comparator.getComparisonCount()
                + ". Should be 8.", 8, comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void rabinKarpSamePattern() {
        /*
            pattern: aeaoahol
            text: aeaoahol
            indices: 0
            expected total comparison: 8
         */
        assertEquals(samePatternAnswer,
                PatternMatching.rabinKarp(samePattern, samePattern, comparator));
        assertTrue("Did not use the comparator.",
                comparator.getComparisonCount() != 0);
        assertEquals("Comparison count was " + comparator.getComparisonCount()
                + ". Should be 8.", 8, comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void kmpMultiPattern1() {
        /*
            pattern: 12312
            text: 1234512312331212312451312312
            indices: 5, 14, 23
            expected total comparison: 37
         */
        assertEquals(multiplePattern1Answer,
                PatternMatching.kmp(multiPattern1, multiPattern1Text, comparator));
        assertTrue("Did not use the comparator.",
                comparator.getComparisonCount() != 0);
        assertEquals("Comparison count was " + comparator.getComparisonCount()
                + ". Should be 37.", 37, comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void boyerMooreMultiPattern1() {
        /*
            pattern: 12312
            text: 1234512312331212312451312312
            indices: 5, 14, 23
            expected total comparison: 33
         */
        assertEquals(multiplePattern1Answer,
                PatternMatching.boyerMoore(multiPattern1, multiPattern1Text, comparator));
        assertTrue("Did not use the comparator.",
                comparator.getComparisonCount() != 0);
        assertEquals("Comparison count was " + comparator.getComparisonCount()
                + ". Should be 33.", 33, comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void rabinKarpMultiPattern1() {
        /*
            pattern: 12312
            text: 1234512312331212312451312312
            indices: 5, 14, 23
            expected total comparison: 15
         */
        assertEquals(multiplePattern1Answer,
                PatternMatching.rabinKarp(multiPattern1, multiPattern1Text, comparator));
        assertTrue("Did not use the comparator.",
                comparator.getComparisonCount() != 0);
        assertEquals("Comparison count was " + comparator.getComparisonCount()
                + ". Should be 15.", 15, comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void kmpAllMatch() {
        /*
            pattern: aba
            text: ababababababababababababa
            indices: 0, 2, 4...22
            expected total comparison: 27
         */
        assertEquals(allMatchPatternAnswer,
                PatternMatching.kmp(allMatchPattern, allMatchPatternText, comparator));
        assertTrue("Did not use the comparator.",
                comparator.getComparisonCount() != 0);
        assertEquals("Comparison count was " + comparator.getComparisonCount()
                + ". Should be 27.", 27, comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void boyerMooreAllMatch() {
        /*
            pattern: aba
            text: ababababababababababababa
            indices: 0, 2, 4...22
            expected total comparison: 47
         */
        assertEquals(allMatchPatternAnswer,
                PatternMatching.boyerMoore(allMatchPattern, allMatchPatternText, comparator));
        assertTrue("Did not use the comparator.",
                comparator.getComparisonCount() != 0);
        assertEquals("Comparison count was " + comparator.getComparisonCount()
                + ". Should be 47.", 47, comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void rabinKarpAllMatch() {
        /*
            pattern: aba
            text: ababababababababababababa
            indices: 0, 2, 4...22
            expected total comparison: 36
         */
        assertEquals(allMatchPatternAnswer,
                PatternMatching.rabinKarp(allMatchPattern, allMatchPatternText, comparator));
        assertTrue("Did not use the comparator.",
                comparator.getComparisonCount() != 0);
        assertEquals("Comparison count was " + comparator.getComparisonCount()
                + ". Should be 36.", 36, comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void kmpMultiPattern2() {
        /*
            pattern: 1343412
            text: 1343412012134420134341211340341134341221340134341293413134
            indices: 0, 16, 31, 43
            expected total comparison: 65
         */
        assertEquals(multiplePattern2Answer,
                PatternMatching.kmp(multiPattern2, multiPattern2Text, comparator));
        assertTrue("Did not use the comparator.",
                comparator.getComparisonCount() != 0);
        assertEquals("Comparison count was " + comparator.getComparisonCount()
                + ". Should be 65.", 65, comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void boyerMooreMultiPattern2() {
        /*
            pattern: 1343412
            text: 1343412012134420134341211340341134341221340134341293413134
            indices: 0, 16, 31, 43
            expected total comparison: 48
         */
        assertEquals(multiplePattern2Answer,
                PatternMatching.boyerMoore(multiPattern2, multiPattern2Text, comparator));
        assertTrue("Did not use the comparator.",
                comparator.getComparisonCount() != 0);
        assertEquals("Comparison count was " + comparator.getComparisonCount()
                + ". Should be 48.", 48, comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void rabinKarpMultiPattern2() {
        /*
            pattern: 1343412
            text: 1343412012134420134341211340341134341221340134341293413134
            indices: 0, 16, 31, 43
            expected total comparison: 28
         */
        assertEquals(multiplePattern2Answer,
                PatternMatching.rabinKarp(multiPattern2, multiPattern2Text, comparator));
        assertTrue("Did not use the comparator.",
                comparator.getComparisonCount() != 0);
        assertEquals("Comparison count was " + comparator.getComparisonCount()
                + ". Should be 28.", 28, comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void kmpMultiPattern3() {
        /*
            pattern: aaaah
            text: haaaaaaaaaaaaaaaaaaahaaaaaaaaaaaaaaaaaaaah
            indices: 16, 37
            expected total comparison: 80
         */
        assertEquals(multiplePattern3Answer,
                PatternMatching.kmp(multiPattern3, multiPattern3Text, comparator));
        assertTrue("Did not use the comparator.",
                comparator.getComparisonCount() != 0);
        assertEquals("Comparison count was " + comparator.getComparisonCount()
                + ". Should be 80.", 80, comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void boyerMooreMultiPattern3() {
        /*
            pattern: aaaah
            text: haaaaaaaaaaaaaaaaaaahaaaaaaaaaaaaaaaaaaaah
            indices: 16, 37
            expected total comparison: 46
         */
        assertEquals(multiplePattern3Answer,
                PatternMatching.boyerMoore(multiPattern3, multiPattern3Text, comparator));
        assertTrue("Did not use the comparator.",
                comparator.getComparisonCount() != 0);
        assertEquals("Comparison count was " + comparator.getComparisonCount()
                + ". Should be 46.", 46, comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void rabinKarpMultiPattern3() {
        /*
            pattern: aaaah
            text: haaaaaaaaaaaaaaaaaaahaaaaaaaaaaaaaaaaaaaah
            indices: 16, 37
            expected total comparison: 10
         */
        assertEquals(multiplePattern3Answer,
                PatternMatching.rabinKarp(multiPattern3, multiPattern3Text, comparator));
        assertTrue("Did not use the comparator.",
                comparator.getComparisonCount() != 0);
        assertEquals("Comparison count was " + comparator.getComparisonCount()
                + ". Should be 10.", 10, comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void kmpExtremePattern1() {
        /*
            pattern: 1123453145132413
            text: Randomly generated (but very long)
            indices: 0, 40, 80, 120...4000
         */
        assertEquals(extremePattern1Answer,
                PatternMatching.kmp(extremePattern1, extremePattern1Text, comparator));
        assertTrue("Did not use the comparator.",
                comparator.getComparisonCount() != 0);
    }

    @Test(timeout = TIMEOUT)
    public void boyerMooreExtremePattern1() {
        /*
            pattern: 1123453145132413
            text: Randomly generated (but very long)
            indices: 0, 40, 80, 120...4000
         */
        assertEquals(extremePattern1Answer,
                PatternMatching.boyerMoore(extremePattern1, extremePattern1Text, comparator));
        assertTrue("Did not use the comparator.",
                comparator.getComparisonCount() != 0);
    }

    @Test(timeout = TIMEOUT)
    public void rabinKarpExtremePattern1() {
        /*
            pattern: 1123453145132413
            text: Randomly generated (but very long)
            indices: 0, 40, 80, 120...4000
         */
        assertEquals(extremePattern1Answer,
                PatternMatching.rabinKarp(extremePattern1, extremePattern1Text, comparator));
        assertTrue("Did not use the comparator.",
                comparator.getComparisonCount() != 0);
    }

    @Test(timeout = TIMEOUT)
    public void kmpExtremePattern2() {
        /*
            pattern: aababcabcdabcdeabcdef
            text: Randomly generated (but very long)
            indices: 0, 48, 96, 144...2400
         */
        assertEquals(extremePattern2Answer,
                PatternMatching.kmp(extremePattern2, extremePattern2Text, comparator));
        assertTrue("Did not use the comparator.",
                comparator.getComparisonCount() != 0);
    }

    @Test(timeout = TIMEOUT)
    public void boyerMooreExtremePattern2() {
        /*
            pattern: aababcabcdabcdeabcdef
            text: Randomly generated (but very long)
            indices: 0, 48, 96, 144...2400
         */
        assertEquals(extremePattern2Answer,
                PatternMatching.boyerMoore(extremePattern2, extremePattern2Text, comparator));
        assertTrue("Did not use the comparator.",
                comparator.getComparisonCount() != 0);
    }

    @Test(timeout = TIMEOUT)
    public void rabinKarpExtremePattern2() {
        /*
            pattern: aababcabcdabcdeabcdef
            text: Randomly generated (but very long)
            indices: 0, 48, 96, 144...2400
         */
        assertEquals(extremePattern2Answer,
                PatternMatching.rabinKarp(extremePattern2, extremePattern2Text, comparator));
        assertTrue("Did not use the comparator.",
                comparator.getComparisonCount() != 0);
    }
}
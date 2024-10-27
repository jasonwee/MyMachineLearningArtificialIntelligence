package ch.weetech.vector;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

/*
 * https://inside.java/2024/10/23/java-and-ai/
 */
public class FFMArrayAddition {
	  public static void main(String[] args) {
	        int arraySize = 1000000;
	        try (Arena arena = Arena.ofConfined()) {
	            // Allocate memory for the arrays
	            MemorySegment array1 = arena.allocate(arraySize * ValueLayout.JAVA_INT.byteSize());
	            MemorySegment array2 = arena.allocate(arraySize * ValueLayout.JAVA_INT.byteSize());
	            MemorySegment result = arena.allocate(arraySize * ValueLayout.JAVA_INT.byteSize());

	            // Initialize arrays
	            for (int i = 0; i < arraySize; i++) {
	                array1.setAtIndex(ValueLayout.JAVA_INT, i, i);
	                array2.setAtIndex(ValueLayout.JAVA_INT, i, i * 2);
	            }

	            // Perform addition, which could be converted to a kernel function to be
	            // executed on a GPU.
	            for (int i = 0; i < arraySize; i++) {
	                int sum = array1.getAtIndex(ValueLayout.JAVA_INT, i) +
	                          array2.getAtIndex(ValueLayout.JAVA_INT, i);
	                result.setAtIndex(ValueLayout.JAVA_INT, i, sum);
	            }
	        }
	    }
}

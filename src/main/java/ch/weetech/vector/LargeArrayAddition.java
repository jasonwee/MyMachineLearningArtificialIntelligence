package ch.weetech.vector;

/*
 * https://inside.java/2024/10/23/java-and-ai/
 */
public class LargeArrayAddition {
	 public static void main(String[] args) {
	        // Two large integer arrays - array1 and array2
	        int[] array1 = new int[1000000];
	        int[] array2 = new int[1000000];

	        // Initialize arrays with values
	        for (int i = 0; i < array1.length; i++) {
	            array1[i] = i;
	            array2[i] = i * 2;
	        }

	        // Output array to store the result
	        int[] result = new int[array1.length];

	        // Add corresponding elements of array1 and array2
	        for (int i = 0; i < array1.length; i++) {
	            result[i] = array1[i] + array2[i];
	        }

	        // Print the result
	        System.out.println("Sum of the first elements: " + result[0]);
	        System.out.println("Sum of the last elements: " + result[result.length - 1]);
	    }
}

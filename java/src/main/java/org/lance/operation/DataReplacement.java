/*
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.lance.operation;

import org.lance.fragment.DataFile;

import com.google.common.base.MoreObjects;

import java.util.Arrays;
import java.util.List;
import java.util.Objects;

/**
 * Replace data in a column in the dataset with new data. This is used for null column population
 * where we replace an entirely null column with a new column that has data.
 *
 * <p>This operation will only allow replacing files that contain the same schema e.g. if the
 * original files contain columns A, B, C and the new files contain only columns A, B then the
 * operation is not allowed.
 *
 * <p>Corollary to the above: the operation will also not allow replacing files unless the affected
 * columns all have the same datafile layout across the fragments being replaced.
 */
public class DataReplacement implements Operation {
  private final List<DataReplacementGroup> replacements;

  private DataReplacement(List<DataReplacementGroup> replacements) {
    this.replacements = replacements;
  }

  /**
   * Get the list of data replacement groups.
   *
   * @return the list of data replacement groups
   */
  public List<DataReplacementGroup> replacements() {
    return replacements;
  }

  @Override
  public String name() {
    return "DataReplacement";
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(this).add("replacements", replacements).toString();
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) return true;
    if (o == null || getClass() != o.getClass()) return false;
    DataReplacement that = (DataReplacement) o;
    return Objects.equals(replacements, that.replacements);
  }

  /**
   * Create a new builder for DataReplacement.
   *
   * @return a new builder
   */
  public static Builder builder() {
    return new Builder();
  }

  /** Builder for DataReplacement. */
  public static class Builder {
    private List<DataReplacementGroup> replacements;

    public Builder() {}

    /**
     * Set the list of data replacement groups.
     *
     * @param replacements the list of data replacement groups
     * @return this builder
     */
    public Builder replacements(List<DataReplacementGroup> replacements) {
      this.replacements = replacements;
      return this;
    }

    /**
     * Build a new DataReplacement.
     *
     * @return a new DataReplacement
     */
    public DataReplacement build() {
      return new DataReplacement(replacements);
    }
  }

  /** A group of data replacement, containing a fragment ID and a new data file. */
  public static class DataReplacementGroup {
    private final long fragmentId;
    private final DataFile replacedFile;
    private final long[] dependencyFieldIds;
    private final long[] mutatedOffsets;

    /**
     * Create a new DataReplacementGroup that declares no dependencies and no mutated offsets: it
     * conflicts on the fields it writes and stamps every row in the fragment as updated.
     *
     * @param fragmentId the fragment ID
     * @param replacedFile the new data file to replace old file of the fragment id
     */
    public DataReplacementGroup(long fragmentId, DataFile replacedFile) {
      this(fragmentId, replacedFile, new long[0], null);
    }

    /**
     * Create a new DataReplacementGroup.
     *
     * @param fragmentId the fragment ID
     * @param replacedFile the new data file to replace old file of the fragment id
     * @param dependencyFieldIds field IDs on this fragment whose values were read to compute the
     *     replaced file. A concurrent transaction that rewrote any of them conflicts with this
     *     replacement. Empty declares no inputs, so the replacement conflicts only on the fields it
     *     writes.
     * @param mutatedOffsets physical row offsets within the fragment whose values this replacement
     *     wrote; only those rows are stamped as updated at commit. Null stamps every row in the
     *     fragment.
     */
    public DataReplacementGroup(
        long fragmentId, DataFile replacedFile, long[] dependencyFieldIds, long[] mutatedOffsets) {
      this.fragmentId = fragmentId;
      this.replacedFile = replacedFile;
      this.dependencyFieldIds = dependencyFieldIds;
      this.mutatedOffsets = mutatedOffsets;
    }

    /**
     * Get the fragment ID.
     *
     * @return the fragment ID
     */
    public long fragmentId() {
      return fragmentId;
    }

    /**
     * Get the new data file.
     *
     * @return the new data file
     */
    public DataFile replacedFile() {
      return replacedFile;
    }

    /**
     * Get the field IDs whose values this replacement was computed from.
     *
     * @return the dependency field IDs, empty when none are declared
     */
    public long[] dependencyFieldIds() {
      return dependencyFieldIds;
    }

    /**
     * Get the physical row offsets this replacement wrote.
     *
     * @return the mutated offsets, or null when the whole fragment is stamped
     */
    public long[] mutatedOffsets() {
      return mutatedOffsets;
    }

    @Override
    public boolean equals(Object o) {
      if (this == o) return true;
      if (o == null || getClass() != o.getClass()) return false;
      DataReplacementGroup that = (DataReplacementGroup) o;
      return fragmentId == that.fragmentId
          && Objects.equals(replacedFile, that.replacedFile)
          && Arrays.equals(dependencyFieldIds, that.dependencyFieldIds)
          && Arrays.equals(mutatedOffsets, that.mutatedOffsets);
    }

    @Override
    public int hashCode() {
      return Objects.hash(
          fragmentId,
          replacedFile,
          Arrays.hashCode(dependencyFieldIds),
          Arrays.hashCode(mutatedOffsets));
    }

    @Override
    public String toString() {
      return MoreObjects.toStringHelper(this)
          .add("fragmentId", fragmentId)
          .add("replacedFile", replacedFile)
          .add("dependencyFieldIds", Arrays.toString(dependencyFieldIds))
          .add("mutatedOffsets", Arrays.toString(mutatedOffsets))
          .toString();
    }
  }
}

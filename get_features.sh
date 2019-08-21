#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
# ----------------------------------------------------------------------------
#
# This script generates a COLMAP reconstruction from a numbe rof input imagess
# Usage: sh get_colmap_reconstruction.sh <COLMAP-exe-directory> <image-set-directory> <project-directory>


# /home/rranftl/Data2/Projects/sfm_priors/colmap

colmap_folder=$1/
iname=$2/
outf=$3/

DATABASE=${outf}/reconstruction.db

PROJECT_PATH=${outf}
IMAGE_PATH=${iname}
mkdir -p ${PROJECT_PATH}

#cp -n ${iname}*.jpg ${PROJECT_PATH}

${colmap_folder}/colmap feature_extractor \
    --database_path ${DATABASE} \
    --image_path ${IMAGE_PATH} \
        --ImageReader.camera_model RADIAL \
        --ImageReader.single_camera 1 \
        --SiftExtraction.use_gpu 1 \
        --SiftExtraction.estimate_affine_shape 0 \


${colmap_folder}/colmap exhaustive_matcher \
    --database_path ${DATABASE} \
    --SiftMatching.use_gpu 1 \
    --SiftMatching.cross_check 0

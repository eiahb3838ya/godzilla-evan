#!/bin/bash

# Set the variables for the filenames
GZ_VERSION_PATH="$(dirname "${1}")"
OLD_NAME="${1%.tar.gz}"
NEW_NAME="godzilla"
FILEPATH="/opt/app/"
WORKING_DIR=${PWD}


# Set the regex pattern to match filenames
pattern='.*/[0-9]{4}\.[0-9]{1,2}\.[0-9]{1,2}-[a-zA-Z0-9]*'
# Set the directory to search for subdirectories
gz_directory="${FILEPATH}godzilla/"

# Search for all subdirectories in the directory with names matching the regex
subdirs=$(find "$gz_directory" -maxdepth 1 -type d -regextype posix-extended -regex "${pattern}")

# If no subdirectories are found, exit with an error message
if [ -z "$subdirs" ]; then
  echo "No subdirectories found in directory '$directory' with the name '[0-9]{4}\.[0-9]{1,2}\.[0-9]{1,2}-[a-zA-Z0-9]*'"
fi

# Get the first subdirectory found and strip the directory path prefix
FILENAME=${subdirs##*${gz_directory}}

# Print the name of the first subdirectory found without the directory path prefix
echo "First subdirectory found in directory '$directory' with the name '[0-9]{4}\.[0-9]{1,2}\.[0-9]{1,2}-[a-zA-Z0-9]*':"
echo "$FILENAME"

# Extract the contents of the "${OLD_NAME}.tar.gz" archive
tar -xvf "${OLD_NAME}"".tar.gz" -C "${GZ_VERSION_PATH}"
# Copy the extracted directory to "/opt/app/"
cp -r "${OLD_NAME}" "${FILEPATH}"
# Print the name of the first subdirectory found without the directory path prefix
echo "First subdirectory found in directory '$directory' with the name '[0-9]{4}\.[0-9]{1,2}\.[0-9]{1,2}-[a-zA-Z0-9]*':"
echo "$FILENAME"

echo "renaming current godzilla ""${NEW_NAME}" " to rollback ""${FILEPATH}""godzilla_rollback""_""${FILENAME}"
# Rename the existing directory to "godzilla_rollback"
if [ -d "${FILEPATH}${NEW_NAME}" ]; then
  # Check if ${FILEPATH}godzilla_rollback_${FILENAME} exists
  if [ -d "${FILEPATH}godzilla_rollback_${FILENAME}" ]; then
    read -p "The roll back: "${FILEPATH}godzilla_rollback_${FILENAME}" already exists. Do you want to replace it? (y/n) " confirm
    if [ "${confirm}" == "y" ]; then
        rm -r "${FILEPATH}godzilla_rollback_${FILENAME}"
        mv -f "${FILEPATH}${NEW_NAME}" "${FILEPATH}godzilla_rollback_${FILENAME}"
    else
      echo "Operation cancelled."
      return 0
    fi
  else
    mv -f "${FILEPATH}${NEW_NAME}" "${FILEPATH}godzilla_rollback_${FILENAME}"
  fi
else
  echo "Source directory does not exist. Proceed to make new"
fi
echo "change " "${GZ_VERSION_PATH}""${OLD_NAME}" " to ""${FILEPATH}""${NEW_NAME}"
# Rename the extracted directory to "godzilla"

mv -f "${OLD_NAME}" "${FILEPATH}""${NEW_NAME}"
echo "returning back to ${WORKING_DIR}"
cd ${WORKING_DIR}


#!/bin/bash

program_name="$0"
identifiers="$@"
if ! which aws >/dev/null; then
    echo "Command 'aws' not found"
    echo
    echo Please install the AWS CLI: http://docs.aws.amazon.com/cli/latest/userguide/installing.html
    exit 1
fi

if ! which jq >/dev/null; then
    echo "Command 'jq' not found"
    echo
    echo Please install jq: https://stedolan.github.io/jq/download/
    exit 1
fi

BEAKER_STAGE=${BEAKER_STAGE:-"prod"}
if [ "$BEAKER_STAGE" = "prod" ]; then
    BEAKER_URL_PREFIX="http://beaker.allenai.org"
    S3_URL_PREFIX="s3://ai2-beaker/prod"
elif [ "$BEAKER_STAGE" = "dev" ]; then
    BEAKER_URL_PREFIX="http://beaker.dev.allenai.org"
    S3_URL_PREFIX="s3://ai2-beaker/dev"
else
    echo "Unrecognized beaker stage provided."
    exit 1
fi

fetch() {
    ex_id=$1
    group_name=$2

    if [ "" = "$group_name" ]; then
        # We aren't fetching a group, so just fetch directly to the experiment_id.
        destination="beaker-early-results/$ex_id/"
        echo "Fetching early results from experiment $BEAKER_URL_PREFIX/ex/$ex_id into $destination"
    else
        # fetch to the group_name, with spaces replaced with underscores.
        destination="beaker-early-results/${group_name// /_}/$ex_id/"
        echo "Fetching early results from experiment $BEAKER_URL_PREFIX/ex/$ex_id (in group $group_name) into $destination"
    fi

    mkdir -p $destination
    aws s3 sync "$S3_URL_PREFIX/early-results/$ex_id/" "$destination" --delete
}

if [ "" = "$identifiers" ]; then
    echo "Please provide experiment ids or group names (or both)."
    echo
    echo Examples:
    echo
    echo $program_name ex_abc123def456 ex_ghi789jkl012
    echo $program_name ex_abc123def456 ex_ghi789jkl012 my_group_name
    echo $program_name my_group_name
    exit 1
fi

for candidate_id in "$identifiers"; do
    if (echo $candidate_id | grep -q '^ex_'); then
        echo "Fetching Experiment ..."
        fetch $candidate_id
    else
        echo "Fetching Group ..."
        # Group names can have spaces in, which we need to
        # replace with something that curl can handle.
        space_escaped_candidate_id=${candidate_id// /%20}
        group_response=$(curl -s "$BEAKER_URL_PREFIX/api/groups/$space_escaped_candidate_id")
        if (echo $group_response | grep -q 'Group not found'); then
            echo $group_response
        else
            for ex_id in $(echo $group_response | jq -r '.experiment_ids[]'); do
                fetch $ex_id "$candidate_id"
            done
        fi
    fi
done

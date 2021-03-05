#!/bin/sh
#
# Test SKR_Web_API examples
#

BASEDIR=$(dirname $0)

CP=$BASEDIR/lib/httpclient-4.3.6.jar:$BASEDIR/lib/httpclient-cache-4.3.6.jar
CP=$CP:$BASEDIR/lib/httpcore-4.3.3.jar:$BASEDIR/lib/fluent-hc-4.3.6.jar
CP=$CP:$BASEDIR/lib/httpmime-4.3.6.jar:$BASEDIR/lib/commons-codec-1.6.jar
CP=$CP:$BASEDIR/lib/commons-logging-1.1.3.jar
CP=$CP:$BASEDIR/lib/skrAPI.jar

# CLASSES=( GenericBatchTest MMInteractiveTest MMInteractiveUserTest SRInteractiveTest )
for mainclass in GenericBatchTest MMInteractiveEnv MMInteractiveUserEnv SRInteractiveEnv; do
    echo "-------- running $mainclass --------"
    java -cp $BASEDIR/classes:$CP $mainclass
done
for mainclass in GenericBatchNew GenericBatchUser; do
    echo "-------- running $mainclass --------"
        java -cp $BASEDIR/classes:$CP $mainclass --email $EMAIL sample.txt
done
exit 0
